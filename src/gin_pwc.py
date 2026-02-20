"""GIN backbone + pairwise classifier heads for algorithm selection (GIN-PWC)."""

from __future__ import annotations

import warnings

# Suppress deprecation warnings from PyTorch Geometric / torch (library code, not ours)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*torch_geometric\.distributed.*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*torch\.jit\.script.*deprecated.*",
)

import json
import logging
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_mean_pool

from .gin_ehm import (
    NodeVocabulary,
    build_vocabulary_from_graph_dicts,
    graph_dict_to_gin_data,
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


def num_pairs(K: int) -> int:
    """Number of unordered pairs (i,j) with 0 <= i < j < K."""
    return K * (K - 1) // 2


def pair_to_idx(i: int, j: int, K: int) -> int:
    """Index of pair (i, j) with 0 <= i < j < K. Order: (0,1), (0,2), ..., (0,K-1), (1,2), ..., (K-2, K-1)."""
    if not (0 <= i < j < K):
        raise ValueError(f"Invalid pair ({i}, {j}) for K={K}")
    return i * (2 * K - i - 1) // 2 + (j - i - 1)


def idx_to_pair(idx: int, K: int) -> tuple[int, int]:
    """Return (i, j) for the given pair index."""
    n = num_pairs(K)
    if not (0 <= idx < n):
        raise ValueError(f"Invalid pair index {idx} for K={K} (num_pairs={n})")
    # Find i: start_i = i*(2*K-i-1)//2
    i = 0
    while i < K - 1:
        start_next = (i + 1) * (2 * K - (i + 1) - 1) // 2
        if idx < start_next:
            break
        i += 1
    j = i + 1 + (idx - i * (2 * K - i - 1) // 2)
    return (i, j)


def all_pairs(K: int) -> list[tuple[int, int]]:
    """List of (i, j) in the same order as pair indices 0 .. num_pairs(K)-1."""
    return [idx_to_pair(p, K) for p in range(num_pairs(K))]


class GINPwcBackbone(nn.Module):
    """
    GIN backbone: embed -> GIN convs -> global mean pool -> graph embedding g.
    Same structure as the non-head part of GINMultiHeadEHM.
    """

    def __init__(
        self,
        num_node_types: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_node_types + 1, hidden_dim, padding_idx=0)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(nn=mlp, train_eps=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # x: (total_nodes, 1) long
        h = self.embed(x.squeeze(-1))
        for conv in self.convs:
            h = conv(h, edge_index).relu()
            h = self.dropout(h)
        g = global_mean_pool(h, batch)  # (batch_size, hidden_dim)
        return g

    def forward_data(self, data: Batch) -> torch.Tensor:
        """Forward from a PyG Batch; returns (batch_size, hidden_dim)."""
        return self.forward(data.x, data.edge_index, data.batch)


class GINPwc(nn.Module):
    """
    GIN backbone + one binary head per solver pair (i,j), i < j.
    Each head maps graph embedding g -> logit for "solver i better than solver j".
    """

    def __init__(
        self,
        num_node_types: int,
        num_solvers: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_solvers = num_solvers
        self._num_pairs = num_pairs(num_solvers)
        self.backbone = GINPwcBackbone(
            num_node_types=num_node_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self._num_pairs)
            ]
        )

    def forward_embedding(self, data: Batch) -> torch.Tensor:
        """Return graph embeddings (batch_size, hidden_dim)."""
        return self.backbone.forward_data(data)

    def forward_all_pairs(self, g: torch.Tensor) -> torch.Tensor:
        """Given g (batch_size, hidden_dim), return logits (batch_size, num_pairs)."""
        return torch.cat([head(g) for head in self.heads], dim=1)

    def forward_data(self, data: Batch) -> torch.Tensor:
        """Full forward: (batch_size, num_pairs) logits."""
        g = self.forward_embedding(data)
        return self.forward_all_pairs(g)

    def forward_batch_for_loss(
        self,
        data: Batch,
        pair_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        For a batch of (graph, pair_index), return logits (batch_size,) for the
        corresponding head per sample. Used in weighted BCE training.
        """
        g = self.forward_embedding(data)  # (B, H)
        logits_all = self.forward_all_pairs(g)  # (B, num_pairs)
        B = g.size(0)
        logits = logits_all[torch.arange(B, device=g.device), pair_idx]
        return logits


class GINPwcSelector(SolverSelector):
    """
    Algorithm selector using GIN-PWC: shared GIN backbone + pairwise binary heads.
    Inference: run all pair heads on graph embedding -> votes per solver -> return argmax votes (tie-break by random).
    """

    def __init__(
        self,
        model: GINPwc,
        vocabulary: NodeVocabulary,
        fallback_solver_ids: list[int],
        graph_timeout: int,
        device: str | None = None,
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.fallback_solver_ids = list(fallback_solver_ids)
        self.graph_timeout = graph_timeout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed
        self.model.to(self.device)
        self.model.eval()
        self._K = model.num_solvers
        self._pair_order = all_pairs(self._K)

    def _get_rank_lst(self, logits: torch.Tensor) -> list[int]:
        """Given logits (num_pairs,) for all pairs, return solver indices ranked by votes (best first)."""
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
        order = sorted(range(self._K), key=lambda k: rec[k], reverse=True)
        return order

    def algorithm_select(self, instance_path: str | Path) -> int:
        path = Path(instance_path)
        graph_dict = build_smt_graph_dict_timeout(path, self.graph_timeout)
        if graph_dict is None:
            _suppress_z3_destructor_noise()
            return self.fallback_solver_ids[0]
        data = graph_dict_to_gin_data(graph_dict, self.vocabulary)
        if data is None:
            _suppress_z3_destructor_noise()
            return self.fallback_solver_ids[0]
        batch = Batch.from_data_list([data])
        batch = batch.to(self.device)
        with torch.no_grad():
            logits = self.model.forward_data(batch)  # (1, num_pairs)
        rank = self._get_rank_lst(logits[0])
        _suppress_z3_destructor_noise()
        return rank[0]

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "num_node_types": self.model.backbone.embed.num_embeddings - 1,
            "num_solvers": self._K,
            "hidden_dim": self.model.backbone.embed.embedding_dim,
            "num_layers": len(self.model.backbone.convs),
            "graph_timeout": self.graph_timeout,
            "fallback_solver_ids": self.fallback_solver_ids,
            "random_seed": self.random_seed,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        with open(save_dir / "vocab.json", "w") as f:
            json.dump({"type_names": self.vocabulary.type_names()}, f)
        logging.info("Saved GIN-PWC model to %s", save_dir)

    @staticmethod
    def load(load_path: str | Path, device: str | None = None) -> "GINPwcSelector":
        load_path = Path(load_path)
        with open(load_path / "config.json") as f:
            config = json.load(f)
        with open(load_path / "vocab.json") as vf:
            vocab_data = json.load(vf)
        type_names = vocab_data["type_names"]
        vocabulary = NodeVocabulary()
        for t in type_names:
            vocabulary.add_type(t)
        vocabulary.freeze()

        model = GINPwc(
            num_node_types=config["num_node_types"],
            num_solvers=config["num_solvers"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=0.0,
        )
        model.load_state_dict(
            torch.load(load_path / "model.pt", map_location="cpu", weights_only=True)
        )
        fallback_solver_ids = config.get(
            "fallback_solver_ids", config.get("timeout_solver_ids", [])
        )
        graph_timeout = config["graph_timeout"]
        random_seed = config.get("random_seed", 42)
        return GINPwcSelector(
            model=model,
            vocabulary=vocabulary,
            fallback_solver_ids=fallback_solver_ids,
            graph_timeout=graph_timeout,
            device=device,
            random_seed=random_seed,
        )


def build_gin_pwc_samples(
    instance_paths: list[str],
    graph_dict_by_path: dict[str, dict],
    multi_perf_data: MultiSolverDataset,
    vocabulary: NodeVocabulary,
) -> list[tuple[Data, int, int, float]]:
    """
    Build list of (Data, pair_idx, label, weight) for GIN-PWC training.
    Only includes (instance, pair) with |PAR2_i - PAR2_j| > PERF_DIFF_THRESHOLD.
    label = 1 if solver i better than j else 0; weight = |PAR2_i - PAR2_j|.
    """
    K = multi_perf_data.num_solvers()
    samples: list[tuple[Data, int, int, float]] = []
    for path in instance_paths:
        graph_dict = graph_dict_by_path.get(path)
        if graph_dict is None:
            continue
        data = graph_dict_to_gin_data(graph_dict, vocabulary)
        if data is None:
            logging.debug("Skipping instance with invalid graph (out-of-bounds edges): %s", path)
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
                pair_idx = pair_to_idx(i, j, K)
                samples.append((data, pair_idx, label, cost))
    return samples


class GINPwcDataset(Dataset[tuple[Data, int, int, float]]):
    """Dataset of (Data, pair_idx, label, weight) for weighted BCE training."""

    def __init__(
        self,
        samples: list[tuple[Data, int, int, float]],
    ) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Data, int, int, float]:
        return self.samples[idx]


def _collate_gin_pwc(
    batch: list[tuple[Data, int, int, float]],
) -> tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate (Data, pair_idx, label, weight) list into Batch + stacked tensors."""
    data_list = [b[0] for b in batch]
    pair_indices = torch.tensor([b[1] for b in batch], dtype=torch.long)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    weights = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return (
        Batch.from_data_list(data_list),
        pair_indices,
        labels,
        weights,
    )


def _weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """BCE with logits, weighted by performance difference. Mean over batch (by sum(weights))."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    w_sum = weights.sum().clamp(min=1e-8)
    return (weights * bce).sum() / w_sum


def train_gin_pwc(
    multi_perf_data: MultiSolverDataset,
    save_dir: str | Path,
    *,
    graph_timeout: int = 5,
    jobs: int = 1,
    hidden_dim: int = 64,
    num_layers: int = 3,
    num_epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout: float = 0.1,
    device: str | None = None,
    val_ratio: float = 0.1,
    patience: int = 50,
    val_split_seed: int = 42,
    min_epochs: int = 50,
) -> None:
    """
    Build graphs, vocab, pairwise samples (with weight = |PAR2_i - PAR2_j|);
    train GINPwc with weighted BCE; save model, vocab, config, failed_paths.
    jobs: number of parallel workers for graph building; 1 = sequential.
    If val_ratio > 0 and patience > 0: split off val_ratio of samples for validation,
    stop when validation loss does not improve for patience epochs (after at least min_epochs), and restore best checkpoint.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    instance_paths = list(multi_perf_data.keys())
    K = multi_perf_data.num_solvers()
    logging.info(
        "Building graphs for %d instances (timeout=%ds)...",
        len(instance_paths),
        graph_timeout,
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

    train_paths = list(graph_by_path.keys())
    graph_dicts = [graph_by_path[p] for p in train_paths]
    vocab = build_vocabulary_from_graph_dicts(graph_dicts)
    num_node_types = vocab.num_types()

    samples = build_gin_pwc_samples(
        train_paths, graph_by_path, multi_perf_data, vocab
    )
    if not samples:
        raise ValueError(
            "No pairwise samples with |PAR2_i - PAR2_j| > PERF_DIFF_THRESHOLD. "
            "Check performance data and threshold."
        )
    logging.info(
        "GIN-PWC: %d instances, %d pairwise samples",
        len(train_paths),
        len(samples),
    )
    dataset = GINPwcDataset(samples)
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
            collate_fn=_collate_gin_pwc,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_gin_pwc,
        )
        logging.info(
            "Early stopping: val_ratio=%.2f, patience=%d, min_epochs=%d -> %d train, %d val samples",
            val_ratio,
            patience,
            min_epochs,
            len(train_indices),
            len(val_indices),
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_gin_pwc,
        )
        val_loader = None

    model = GINPwc(
        num_node_types=num_node_types,
        num_solvers=K,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
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
        for batch_data, pair_idx, labels, weights in train_loader:
            batch_data = batch_data.to(device)
            pair_idx = pair_idx.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            logits = model.forward_batch_for_loss(batch_data, pair_idx)
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
                for batch_data, pair_idx, labels, weights in val_loader:
                    batch_data = batch_data.to(device)
                    pair_idx = pair_idx.to(device)
                    labels = labels.to(device)
                    weights = weights.to(device)
                    logits = model.forward_batch_for_loss(batch_data, pair_idx)
                    loss = _weighted_bce_loss(logits, labels, weights)
                    val_loss_sum += loss.item()
                    val_batches += 1
            val_loss = val_loss_sum / val_batches if val_batches else float("inf")
            improved = best_val_loss is None or val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_state_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info(
                    "Epoch %d/%d train loss (weighted BCE) %.4f val loss %.4f%s",
                    epoch + 1,
                    num_epochs,
                    avg_loss,
                    val_loss,
                    " (best)" if improved else "",
                )
            if epochs_no_improve >= patience and (epoch + 1) >= min_epochs:
                logging.info(
                    "Early stop: no val improvement for %d epochs (min_epochs=%d reached), restoring best",
                    patience,
                    min_epochs,
                )
                model.load_state_dict(torch.load(best_state_path, map_location=device, weights_only=True))
                if best_state_path.exists():
                    best_state_path.unlink()
                break
        else:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info(
                    "Epoch %d/%d train loss (weighted BCE) %.4f",
                    epoch + 1,
                    num_epochs,
                    avg_loss,
                )

    if best_state_path.exists():
        model.load_state_dict(torch.load(best_state_path, map_location=device, weights_only=True))
        best_state_path.unlink(missing_ok=True)

    fallback_solver_ids = sorted_fallback_solvers(multi_perf_data, failed_list)
    config = {
        "num_node_types": num_node_types,
        "num_solvers": K,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "graph_timeout": graph_timeout,
        "timeout": multi_perf_data.get_timeout(),
        "fallback_solver_ids": fallback_solver_ids,
        "random_seed": 42,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    torch.save(model.state_dict(), save_dir / "model.pt")
    with open(save_dir / "vocab.json", "w") as f:
        json.dump({"type_names": vocab.type_names()}, f)
    with open(save_dir / "failed_paths.txt", "w") as f:
        for p in failed_list:
            f.write(p + "\n")
    logging.info("Saved GIN-PWC model, vocab, config, failed_paths to %s", save_dir)
