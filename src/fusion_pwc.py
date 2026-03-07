"""Fusion of GIN + text embeddings with pairwise classification (Fusion-PWC)."""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .gin_pwc import (
    all_pairs,
    pair_to_idx,
    _weighted_bce_loss,
)
from .performance import MultiSolverDataset, PERF_DIFF_THRESHOLD
from .solver_selector import SolverSelector

GIN_L2_EPS = 1e-8


def load_embedding_csv(csv_path: Path | str) -> dict[str, np.ndarray]:
    """
    Load path -> embedding vector from a CSV with columns path, emb_0, emb_1, ...
    Returns dict keyed by path string (as in CSV).
    """
    csv_path = Path(csv_path)
    emb_cols: list[str] = []
    rows: list[tuple[str, list[float]]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        emb_cols = [c for c in reader.fieldnames if c.startswith("emb_")]
        emb_cols.sort(key=lambda x: int(x.split("_")[1]))
        for row in reader:
            path = (row.get("path") or "").strip().replace("\\", "/")
            if not path:
                continue
            vec = [float(row[c]) for c in emb_cols]
            rows.append((path, vec))
    return {path: np.array(vec, dtype=np.float32) for path, vec in rows}


def normalize_gin_l2(emb: np.ndarray, eps: float = GIN_L2_EPS) -> np.ndarray:
    """L2-normalize a single embedding vector (in-place possible)."""
    norm = np.linalg.norm(emb)
    if norm < eps:
        return emb  # leave zero vector as-is
    return (emb / norm).astype(np.float32)


def build_emb_by_path(
    gin_by_rel: dict[str, np.ndarray],
    text_by_rel: dict[str, np.ndarray],
    benchmark_root: Path | str,
    paths_full: list[str],
    *,
    normalize_gin: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build path (full) -> (gin_emb, text_emb) for paths in paths_full that have both.
    gin_by_rel, text_by_rel are keyed by relative path (e.g. BV/...).
    """
    root = Path(benchmark_root).resolve()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for full in paths_full:
        try:
            rel = str(Path(full).relative_to(root))
        except ValueError:
            rel = full
        rel = rel.replace("\\", "/")
        g = gin_by_rel.get(rel)
        t = text_by_rel.get(rel)
        if g is None or t is None:
            continue
        if normalize_gin:
            g = normalize_gin_l2(g)
        out[full] = (g, t)
    return out


class FusionPWC(nn.Module):
    """
    Fusion: text 768 -> d_text_small; concat(gin_64, text_small) -> 128;
    LayerNorm; Linear(128, hidden_fused) -> ReLU -> Dropout;
    then one binary head per solver pair (same as GIN-PWC).
    """

    def __init__(
        self,
        d_gin: int,
        d_text: int,
        num_solvers: int,
        d_text_small: int = 64,
        hidden_fused: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_gin = d_gin
        self.d_text = d_text
        self.d_text_small = d_text_small
        self.hidden_fused = hidden_fused
        self._num_solvers = num_solvers
        self._num_pairs = (num_solvers * (num_solvers - 1)) // 2

        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_text_small),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        concat_dim = d_gin + d_text_small
        self.layer_norm = nn.LayerNorm(concat_dim)
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, hidden_fused),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_fused, hidden_fused),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_fused, 1),
                )
                for _ in range(self._num_pairs)
            ]
        )

    def forward_fused(
        self,
        gin: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """(B, d_gin), (B, d_text) -> (B, hidden_fused)."""
        t = self.text_proj(text)
        concat = torch.cat([gin, t], dim=1)
        concat = self.layer_norm(concat)
        return self.proj(concat)

    def forward_all_pairs(self, fused: torch.Tensor) -> torch.Tensor:
        """(B, hidden_fused) -> (B, num_pairs)."""
        return torch.cat([h(fused) for h in self.heads], dim=1)

    def forward_batch_for_loss(
        self,
        gin: torch.Tensor,
        text: torch.Tensor,
        pair_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits (B,) for the corresponding head per sample."""
        fused = self.forward_fused(gin, text)
        logits_all = self.forward_all_pairs(fused)
        B = fused.size(0)
        return logits_all[torch.arange(B, device=fused.device), pair_idx]


class FusionPWCSelector(SolverSelector):
    """
    Selector that uses precomputed (gin, text) embeddings and the fusion+PWC model.
    algorithm_select(instance_path): look up embeddings, run model, return solver.
    """

    def __init__(
        self,
        model: FusionPWC,
        emb_by_path: dict[str, tuple[np.ndarray, np.ndarray]],
        fallback_solver_ids: list[int],
        device: str | None = None,
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.emb_by_path = emb_by_path
        self.fallback_solver_ids = list(fallback_solver_ids)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed
        self.model.to(self.device)
        self.model.eval()
        self._num_solvers = model._num_solvers
        self._pair_order = all_pairs(self._num_solvers)

    def _get_rank_lst(self, logits: torch.Tensor) -> list[int]:
        """Logits (num_pairs,) -> solver indices ranked by votes (best first)."""
        pred = (logits > 0).cpu().numpy()
        K = self._num_solvers
        votes = [0] * K
        for idx, (i, j) in enumerate(self._pair_order):
            if pred[idx]:
                votes[i] += 1
            else:
                votes[j] += 1
        rng = random.Random(self.random_seed)
        tiebreaker = [rng.random() for _ in range(K)]
        rec = list(zip(votes, tiebreaker))
        return sorted(range(K), key=lambda k: rec[k], reverse=True)

    def algorithm_select(self, instance_path: str | Path) -> int:
        selected, _, _ = self.algorithm_select_with_info(instance_path)
        return selected

    def algorithm_select_with_info(
        self, instance_path: str | Path
    ) -> tuple[int, float | None, bool]:
        """Return (solver_id, overhead_sec, feature_fail). overhead is None (no graph build)."""
        path = str(Path(instance_path).resolve())
        entry = self.emb_by_path.get(path)
        if entry is None:
            return (self.fallback_solver_ids[0], None, True)
        gin_np, text_np = entry
        gin = torch.from_numpy(gin_np).float().unsqueeze(0).to(self.device)
        text = torch.from_numpy(text_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model.forward_all_pairs(self.model.forward_fused(gin, text))
        rank = self._get_rank_lst(logits[0])
        return (rank[0], None, False)

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "d_gin": self.model.d_gin,
            "d_text": self.model.d_text,
            "d_text_small": self.model.d_text_small,
            "hidden_fused": self.model.hidden_fused,
            "num_solvers": len(self.fallback_solver_ids),
            "fallback_solver_ids": self.fallback_solver_ids,
            "random_seed": self.random_seed,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        logging.info("Saved Fusion-PWC model to %s", save_dir)

    @staticmethod
    def load(
        load_path: str | Path,
        emb_by_path: dict[str, tuple[np.ndarray, np.ndarray]],
        device: str | None = None,
    ) -> "FusionPWCSelector":
        load_path = Path(load_path)
        with open(load_path / "config.json") as f:
            config = json.load(f)
        model = FusionPWC(
            d_gin=config["d_gin"],
            d_text=config["d_text"],
            num_solvers=config["num_solvers"],
            d_text_small=config["d_text_small"],
            hidden_fused=config["hidden_fused"],
            dropout=0.0,
        )
        model.load_state_dict(
            torch.load(load_path / "model.pt", map_location="cpu", weights_only=True)
        )
        fallback_solver_ids = config.get("fallback_solver_ids", [])
        random_seed = config.get("random_seed", 42)
        return FusionPWCSelector(
            model=model,
            emb_by_path=emb_by_path,
            fallback_solver_ids=fallback_solver_ids,
            device=device,
            random_seed=random_seed,
        )


def build_fusion_pwc_samples(
    instance_paths: list[str],
    emb_by_path: dict[str, tuple[np.ndarray, np.ndarray]],
    multi_perf_data: MultiSolverDataset,
) -> list[tuple[str, int, int, float]]:
    """
    Build list of (path, pair_idx, label, weight) for Fusion-PWC training.
    Only instances in emb_by_path and pairs with |PAR2_i - PAR2_j| > PERF_DIFF_THRESHOLD.
    """
    K = multi_perf_data.num_solvers()
    samples: list[tuple[str, int, int, float]] = []
    for path in instance_paths:
        if path not in emb_by_path:
            continue
        for i in range(K):
            for j in range(i + 1, K):
                par2_i = multi_perf_data.get_par2(path, i)
                par2_j = multi_perf_data.get_par2(path, j)
                if par2_i is None or par2_j is None:
                    continue
                cost = abs(par2_i - par2_j)
                if cost <= PERF_DIFF_THRESHOLD:
                    continue
                label = 1 if par2_i < par2_j else 0
                pair_idx = pair_to_idx(i, j, K)
                samples.append((path, pair_idx, label, cost))
    return samples


class FusionPWCDataset(Dataset[tuple[np.ndarray, np.ndarray, int, int, float]]):
    """Dataset of (gin_emb, text_emb, pair_idx, label, weight) from (path, ...) samples."""

    def __init__(
        self,
        samples: list[tuple[str, int, int, float]],
        emb_by_path: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.samples = samples
        self.emb_by_path = emb_by_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray, int, int, float]:
        path, pair_idx, label, weight = self.samples[idx]
        gin, text = self.emb_by_path[path]
        return (gin, text, pair_idx, label, weight)


def _collate_fusion_pwc(
    batch: list[tuple[np.ndarray, np.ndarray, int, int, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate to (gin, text, pair_idx, labels, weights)."""
    gin_list = [b[0] for b in batch]
    text_list = [b[1] for b in batch]
    pair_indices = torch.tensor([b[2] for b in batch], dtype=torch.long)
    labels = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    weights = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    gin = torch.from_numpy(np.stack(gin_list)).float()
    text = torch.from_numpy(np.stack(text_list)).float()
    return gin, text, pair_indices, labels, weights


def train_fusion_pwc(
    emb_by_path: dict[str, tuple[np.ndarray, np.ndarray]],
    train_paths: list[str],
    multi_perf_data: MultiSolverDataset,
    save_dir: str | Path,
    *,
    d_gin: int = 64,
    d_text: int = 768,
    d_text_small: int = 64,
    hidden_fused: int = 128,
    num_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.1,
    device: str | None = None,
    val_ratio: float = 0.15,
    patience: int = 50,
    val_split_seed: int = 42,
    min_epochs: int = 100,
    failed_paths: list[str] | None = None,
) -> None:
    """
    Train Fusion-PWC on (gin, text) embeddings with weighted BCE.
    failed_paths: paths that failed GIN extraction (for fallback order); if None, use [].
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    failed_list = failed_paths or []

    K = multi_perf_data.num_solvers()
    samples = build_fusion_pwc_samples(train_paths, emb_by_path, multi_perf_data)
    if not samples:
        raise ValueError(
            "No pairwise samples with |PAR2_i - PAR2_j| > PERF_DIFF_THRESHOLD. "
            "Check performance data and embedding coverage."
        )
    logging.info(
        "Fusion-PWC: %d train paths with embeddings, %d pairwise samples",
        len([p for p in train_paths if p in emb_by_path]),
        len(samples),
    )

    dataset = FusionPWCDataset(samples, emb_by_path)
    n_total = len(dataset)
    use_early_stop = val_ratio > 0 and patience > 0 and n_total >= 2

    if use_early_stop:
        rng = random.Random(val_split_seed)
        indices = list(range(n_total))
        rng.shuffle(indices)
        n_val = max(1, int(n_total * val_ratio))
        n_val = min(n_val, n_total - 1)
        val_indices = indices[-n_val:]
        train_indices = indices[:-n_val]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fusion_pwc,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_fusion_pwc,
        )
        logging.info(
            "Early stopping: val_ratio=%.2f, patience=%d, min_epochs=%d -> %d train, %d val",
            val_ratio, patience, min_epochs, len(train_indices), len(val_indices),
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fusion_pwc,
        )
        val_loader = None

    model = FusionPWC(
        d_gin=d_gin,
        d_text=d_text,
        num_solvers=K,
        d_text_small=d_text_small,
        hidden_fused=hidden_fused,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss: float | None = None
    epochs_no_improve = 0
    best_state_path = save_dir / "_best_model.pt"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for gin_b, text_b, pair_idx, labels, weights in train_loader:
            gin_b = gin_b.to(device)
            text_b = text_b.to(device)
            pair_idx = pair_idx.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            logits = model.forward_batch_for_loss(gin_b, text_b, pair_idx)
            loss = _weighted_bce_loss(logits, labels, weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches if n_batches else 0.0

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for gin_b, text_b, pair_idx, labels, weights in val_loader:
                    gin_b = gin_b.to(device)
                    text_b = text_b.to(device)
                    pair_idx = pair_idx.to(device)
                    labels = labels.to(device)
                    weights = weights.to(device)
                    logits = model.forward_batch_for_loss(gin_b, text_b, pair_idx)
                    loss = _weighted_bce_loss(logits, labels, weights)
                    val_loss_sum += loss.item()
                    val_batches += 1
            val_loss = val_loss_sum / val_batches if val_batches else float("inf")
            improved = best_val_loss is None or val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_state_path)
                epochs_no_improve = 0
                logging.info("Best val loss %.4f (epoch %d)", val_loss, epoch + 1)
            else:
                epochs_no_improve += 1
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info(
                    "Epoch %d/%d train loss %.4f val loss %.4f%s",
                    epoch + 1, num_epochs, avg_loss, val_loss, " (best)" if improved else "",
                )
            if epochs_no_improve >= patience and (epoch + 1) >= min_epochs:
                logging.info(
                    "Early stop: no val improvement for %d epochs, restoring best",
                    patience,
                )
                model.load_state_dict(
                    torch.load(best_state_path, map_location=device, weights_only=True)
                )
                if best_state_path.exists():
                    best_state_path.unlink()
                break
        else:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info("Epoch %d/%d train loss %.4f", epoch + 1, num_epochs, avg_loss)

    if best_state_path.exists():
        model.load_state_dict(
            torch.load(best_state_path, map_location=device, weights_only=True)
        )
        best_state_path.unlink(missing_ok=True)

    fallback_solver_ids = [multi_perf_data.get_best_solver_id()]
    config = {
        "d_gin": d_gin,
        "d_text": d_text,
        "d_text_small": d_text_small,
        "hidden_fused": hidden_fused,
        "num_solvers": K,
        "fallback_solver_ids": fallback_solver_ids,
        "random_seed": 42,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    torch.save(model.state_dict(), save_dir / "model.pt")
    if failed_list:
        with open(save_dir / "failed_paths.txt", "w") as f:
            for p in failed_list:
                f.write(p + "\n")
    logging.info("Saved Fusion-PWC model and config to %s", save_dir)
