"""End-to-end fusion: fine-tuned GIN backbone + frozen text embeddings + PWC heads."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch, Data

from .fusion_pwc import load_embedding_csv
from .gin_ehm import (
    NodeVocabulary,
    graph_dict_to_gin_data,
)
from .gin_pwc import (
    GINPwcBackbone,
    all_pairs,
    pair_to_idx,
    _weighted_bce_loss,
)
from .graph_rep import (
    build_smt_graph_dict_timeout,
    generate_graph_dicts,
    generate_graph_dicts_parallel,
    _suppress_z3_destructor_noise,
)
from .performance import MultiSolverDataset, PERF_DIFF_THRESHOLD
from .pwc_wl import sorted_fallback_solvers
from .solver_selector import SolverSelector


class FusionE2EPwc(nn.Module):
    """
    End-to-end fusion: GINPwcBackbone (trainable) + text projection + concat +
    LayerNorm + projection + per-pair PWC binary heads.

    GIN backbone is initialized from a pre-trained GIN-PWC checkpoint.
    Text embeddings are frozen numpy arrays fed in at forward time.
    """

    def __init__(
        self,
        num_node_types: int,
        num_solvers: int,
        d_text: int = 768,
        d_text_small: int = 64,
        hidden_dim: int = 64,
        hidden_fused: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_solvers = num_solvers
        self.d_text = d_text
        self.d_text_small = d_text_small
        self.hidden_dim = hidden_dim
        self.hidden_fused = hidden_fused
        self.num_layers = num_layers
        self._num_pairs = (num_solvers * (num_solvers - 1)) // 2

        self.backbone = GINPwcBackbone(
            num_node_types=num_node_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_text_small),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        concat_dim = hidden_dim + d_text_small
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
        data: Batch,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """(Batch of graphs, (B, d_text)) -> (B, hidden_fused)."""
        g = self.backbone.forward_data(data)
        t = self.text_proj(text)
        concat = torch.cat([g, t], dim=1)
        concat = self.layer_norm(concat)
        return self.proj(concat)

    def forward_all_pairs(self, fused: torch.Tensor) -> torch.Tensor:
        """(B, hidden_fused) -> (B, num_pairs)."""
        return torch.cat([h(fused) for h in self.heads], dim=1)

    def forward_batch_for_loss(
        self,
        data: Batch,
        text: torch.Tensor,
        pair_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits (B,) for the corresponding head per sample."""
        fused = self.forward_fused(data, text)
        logits_all = self.forward_all_pairs(fused)
        B = fused.size(0)
        return logits_all[torch.arange(B, device=fused.device), pair_idx]


def load_gin_pwc_backbone(
    pretrained_dir: Path,
) -> tuple[dict[str, torch.Tensor], NodeVocabulary, dict]:
    """
    Load backbone weights, vocabulary, and config from a trained GIN-PWC model dir.

    Returns (backbone_state_dict, vocabulary, config).
    The backbone_state_dict has keys with the 'backbone.' prefix stripped.
    """
    config_path = pretrained_dir / "config.json"
    model_path = pretrained_dir / "model.pt"
    vocab_path = pretrained_dir / "vocab.json"
    for p in [config_path, model_path, vocab_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found in GIN-PWC model dir: {p}")

    with open(config_path) as f:
        config = json.load(f)
    with open(vocab_path) as f:
        vocab_data = json.load(f)

    vocabulary = NodeVocabulary()
    for t in vocab_data["type_names"]:
        vocabulary.add_type(t)
    vocabulary.freeze()

    full_state = torch.load(model_path, map_location="cpu", weights_only=True)
    prefix = "backbone."
    backbone_state = {
        k[len(prefix):]: v
        for k, v in full_state.items()
        if k.startswith(prefix)
    }
    if not backbone_state:
        raise ValueError(
            f"No 'backbone.*' keys found in GIN-PWC state dict at {model_path}. "
            f"Keys: {list(full_state.keys())[:10]}"
        )
    return backbone_state, vocabulary, config


class FusionE2EPwcSelector(SolverSelector):
    """
    Selector using end-to-end fusion: builds graph on-the-fly, looks up frozen
    text embedding, runs through FusionE2EPwc model, votes across pair heads.
    """

    def __init__(
        self,
        model: FusionE2EPwc,
        vocabulary: NodeVocabulary,
        text_by_path: dict[str, np.ndarray],
        fallback_solver_ids: list[int],
        graph_timeout: int,
        device: str | None = None,
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.text_by_path = text_by_path
        self.fallback_solver_ids = list(fallback_solver_ids)
        self.graph_timeout = graph_timeout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed
        self.model.to(self.device)
        self.model.eval()
        self._K = model.num_solvers
        self._pair_order = all_pairs(self._K)

    def _get_rank_lst(self, logits: torch.Tensor) -> list[int]:
        """Logits (num_pairs,) -> solver indices ranked by votes (best first)."""
        pred = (logits > 0).cpu().numpy()
        votes = [0] * self._K
        for idx, (i, j) in enumerate(self._pair_order):
            if pred[idx]:
                votes[i] += 1
            else:
                votes[j] += 1
        rng = random.Random(self.random_seed)
        tiebreaker = [rng.random() for _ in range(self._K)]
        rec = list(zip(votes, tiebreaker))
        return sorted(range(self._K), key=lambda k: rec[k], reverse=True)

    def algorithm_select(self, instance_path: str | Path) -> int:
        selected, _, _ = self.algorithm_select_with_info(instance_path)
        return selected

    def algorithm_select_with_info(
        self, instance_path: str | Path
    ) -> tuple[int, float, bool]:
        """Return (solver_id, overhead_sec, feature_fail)."""
        path = Path(instance_path)
        full = str(path.resolve())

        text_emb = self.text_by_path.get(full)
        if text_emb is None:
            return (self.fallback_solver_ids[0], 0.0, True)

        t0 = time.perf_counter()
        graph_dict = build_smt_graph_dict_timeout(path, self.graph_timeout)
        t_after_build = time.perf_counter()
        graph_elapsed = t_after_build - t0

        if graph_dict is None:
            _suppress_z3_destructor_noise()
            return (self.fallback_solver_ids[0], min(graph_elapsed, self.graph_timeout), True)

        data = graph_dict_to_gin_data(graph_dict, self.vocabulary)
        if data is None:
            _suppress_z3_destructor_noise()
            overhead = min(graph_elapsed, self.graph_timeout) + (time.perf_counter() - t_after_build)
            return (self.fallback_solver_ids[0], overhead, True)

        batch = Batch.from_data_list([data]).to(self.device)
        text_t = torch.from_numpy(text_emb).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            fused = self.model.forward_fused(batch, text_t)
            logits = self.model.forward_all_pairs(fused)

        rank = self._get_rank_lst(logits[0])
        overhead = min(graph_elapsed, self.graph_timeout) + (time.perf_counter() - t_after_build)
        _suppress_z3_destructor_noise()
        return (rank[0], overhead, False)

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "num_node_types": self.model.backbone.embed.num_embeddings - 1,
            "num_solvers": self._K,
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "d_text": self.model.d_text,
            "d_text_small": self.model.d_text_small,
            "hidden_fused": self.model.hidden_fused,
            "graph_timeout": self.graph_timeout,
            "fallback_solver_ids": self.fallback_solver_ids,
            "random_seed": self.random_seed,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        with open(save_dir / "vocab.json", "w") as f:
            json.dump({"type_names": self.vocabulary.type_names()}, f)
        logging.info("Saved Fusion-E2E-PWC model to %s", save_dir)

    @staticmethod
    def load(
        load_path: str | Path,
        text_by_path: dict[str, np.ndarray],
        device: str | None = None,
    ) -> "FusionE2EPwcSelector":
        load_path = Path(load_path)
        with open(load_path / "config.json") as f:
            config = json.load(f)
        with open(load_path / "vocab.json") as f:
            vocab_data = json.load(f)

        vocabulary = NodeVocabulary()
        for t in vocab_data["type_names"]:
            vocabulary.add_type(t)
        vocabulary.freeze()

        model = FusionE2EPwc(
            num_node_types=config["num_node_types"],
            num_solvers=config["num_solvers"],
            d_text=config["d_text"],
            d_text_small=config["d_text_small"],
            hidden_dim=config["hidden_dim"],
            hidden_fused=config["hidden_fused"],
            num_layers=config["num_layers"],
            dropout=0.0,
        )
        model.load_state_dict(
            torch.load(load_path / "model.pt", map_location="cpu", weights_only=True)
        )
        return FusionE2EPwcSelector(
            model=model,
            vocabulary=vocabulary,
            text_by_path=text_by_path,
            fallback_solver_ids=config.get("fallback_solver_ids", []),
            graph_timeout=config["graph_timeout"],
            device=device,
            random_seed=config.get("random_seed", 42),
        )


def build_fusion_e2e_pwc_samples(
    instance_paths: list[str],
    graph_dict_by_path: dict[str, dict],
    text_by_path: dict[str, np.ndarray],
    multi_perf_data: MultiSolverDataset,
    vocabulary: NodeVocabulary,
) -> list[tuple[Data, np.ndarray, int, int, float]]:
    """
    Build (Data, text_emb, pair_idx, label, weight) samples.
    Only instances present in both graph_dict_by_path and text_by_path,
    and pairs with |PAR2_i - PAR2_j| > PERF_DIFF_THRESHOLD.
    """
    K = multi_perf_data.num_solvers()
    samples: list[tuple[Data, np.ndarray, int, int, float]] = []
    for path in instance_paths:
        graph_dict = graph_dict_by_path.get(path)
        if graph_dict is None:
            continue
        text_emb = text_by_path.get(path)
        if text_emb is None:
            continue
        data = graph_dict_to_gin_data(graph_dict, vocabulary)
        if data is None:
            logging.debug("Skipping instance with invalid graph: %s", path)
            continue
        if data.num_nodes == 0:
            logging.debug("Skipping instance with 0 nodes: %s", path)
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
                pidx = pair_to_idx(i, j, K)
                samples.append((data, text_emb, pidx, label, cost))
    return samples


class FusionE2EPwcDataset(Dataset[tuple[Data, np.ndarray, int, int, float]]):
    """Dataset of (Data, text_emb, pair_idx, label, weight)."""

    def __init__(
        self,
        samples: list[tuple[Data, np.ndarray, int, int, float]],
    ) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Data, np.ndarray, int, int, float]:
        return self.samples[idx]


def _collate_fusion_e2e_pwc(
    batch: list[tuple[Data, np.ndarray, int, int, float]],
) -> tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate to (graph_batch, text, pair_idx, labels, weights)."""
    data_list = [b[0] for b in batch]
    text_list = [b[1] for b in batch]
    pair_indices = torch.tensor([b[2] for b in batch], dtype=torch.long)
    labels = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    weights = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    text = torch.from_numpy(np.stack(text_list)).float()
    return Batch.from_data_list(data_list), text, pair_indices, labels, weights


def train_fusion_e2e_pwc(
    pretrained_gin_dir: str | Path,
    text_by_path: dict[str, np.ndarray],
    multi_perf_data: MultiSolverDataset,
    save_dir: str | Path,
    *,
    graph_timeout: int = 5,
    jobs: int = 1,
    d_text: int = 768,
    d_text_small: int = 64,
    hidden_fused: int = 64,
    num_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.1,
    device: str | None = None,
    val_ratio: float = 0.15,
    patience: int = 50,
    val_split_seed: int = 42,
    min_epochs: int = 100,
) -> None:
    """
    Train Fusion-E2E-PWC: load pre-trained GIN backbone, build graphs,
    fuse with frozen text embeddings, train with weighted BCE + early stopping.
    """
    pretrained_gin_dir = Path(pretrained_gin_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-trained GIN backbone
    backbone_state, vocabulary, gin_config = load_gin_pwc_backbone(pretrained_gin_dir)
    num_node_types = gin_config["num_node_types"]
    hidden_dim = gin_config["hidden_dim"]
    num_layers = gin_config["num_layers"]
    logging.info(
        "Loaded GIN backbone from %s (num_node_types=%d, hidden_dim=%d, num_layers=%d)",
        pretrained_gin_dir, num_node_types, hidden_dim, num_layers,
    )

    # Build graphs
    instance_paths = list(multi_perf_data.keys())
    K = multi_perf_data.num_solvers()
    logging.info(
        "Building graphs for %d instances (timeout=%ds)...",
        len(instance_paths), graph_timeout,
    )
    if jobs > 1:
        graph_by_path, failed_list = generate_graph_dicts_parallel(
            instance_paths, graph_timeout, n_workers=jobs
        )
    else:
        graph_by_path, failed_list = generate_graph_dicts(instance_paths, graph_timeout)
    if not graph_by_path:
        raise ValueError(
            "No graphs could be built. Increase --graph-timeout or check instances."
        )

    # Build samples (require both graph and text embedding)
    train_paths = list(graph_by_path.keys())
    samples = build_fusion_e2e_pwc_samples(
        train_paths, graph_by_path, text_by_path, multi_perf_data, vocabulary
    )
    if not samples:
        raise ValueError(
            "No pairwise samples with |PAR2_i - PAR2_j| > PERF_DIFF_THRESHOLD. "
            "Check performance data, graph coverage, and text embedding coverage."
        )
    n_with_text = sum(1 for p in train_paths if p in text_by_path)
    logging.info(
        "Fusion-E2E-PWC: %d instances with graphs, %d also have text embeddings, %d pairwise samples",
        len(train_paths), n_with_text, len(samples),
    )

    dataset = FusionE2EPwcDataset(samples)
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
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=_collate_fusion_e2e_pwc,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=_collate_fusion_e2e_pwc,
        )
        logging.info(
            "Early stopping: val_ratio=%.2f, patience=%d, min_epochs=%d -> %d train, %d val",
            val_ratio, patience, min_epochs, len(train_indices), len(val_indices),
        )
    else:
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=_collate_fusion_e2e_pwc,
        )
        val_loader = None

    # Build model and load pre-trained backbone
    model = FusionE2EPwc(
        num_node_types=num_node_types,
        num_solvers=K,
        d_text=d_text,
        d_text_small=d_text_small,
        hidden_dim=hidden_dim,
        hidden_fused=hidden_fused,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.backbone.load_state_dict(backbone_state)
    logging.info("Loaded pre-trained GIN backbone weights into FusionE2EPwc")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss: float | None = None
    epochs_no_improve = 0
    best_state_path = save_dir / "_best_model.pt"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_data, text_b, pair_idx, labels, weights in train_loader:
            batch_data = batch_data.to(device)
            text_b = text_b.to(device)
            pair_idx = pair_idx.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            logits = model.forward_batch_for_loss(batch_data, text_b, pair_idx)
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
                for batch_data, text_b, pair_idx, labels, weights in val_loader:
                    batch_data = batch_data.to(device)
                    text_b = text_b.to(device)
                    pair_idx = pair_idx.to(device)
                    labels = labels.to(device)
                    weights = weights.to(device)
                    logits = model.forward_batch_for_loss(batch_data, text_b, pair_idx)
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
                    epoch + 1, num_epochs, avg_loss, val_loss,
                    " (best)" if improved else "",
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

    # Save model, vocab, config, failed_paths
    fallback_solver_ids = sorted_fallback_solvers(multi_perf_data, failed_list)
    config = {
        "num_node_types": num_node_types,
        "num_solvers": K,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "d_text": d_text,
        "d_text_small": d_text_small,
        "hidden_fused": hidden_fused,
        "graph_timeout": graph_timeout,
        "fallback_solver_ids": fallback_solver_ids,
        "random_seed": 42,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    torch.save(model.state_dict(), save_dir / "model.pt")
    with open(save_dir / "vocab.json", "w") as f:
        json.dump({"type_names": vocabulary.type_names()}, f)
    with open(save_dir / "failed_paths.txt", "w") as f:
        for p in failed_list:
            f.write(p + "\n")
    logging.info("Saved Fusion-E2E-PWC model, vocab, config, failed_paths to %s", save_dir)
