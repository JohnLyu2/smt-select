"""GIN empirical hardness model: model, vocab, data conversion, training, and selector."""

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
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_mean_pool

from .graph_rep import (
    smt_graph_to_gin,
    build_smt_graph_dict_timeout,
    generate_graph_dicts,
    generate_graph_dicts_parallel,
    _suppress_z3_destructor_noise,
)
from .performance import MultiSolverDataset
from .pwc_wl import sorted_fallback_solvers
from .solver_selector import SolverSelector

# Index for node types not in the vocabulary (e.g. at inference time).
UNK_TYPE_INDEX = 0


class NodeVocabulary:
    """
    Maps node type strings to integer indices for GIN node features.

    Unknown types map to UNK_TYPE_INDEX (0). Known types get indices 1, 2, ...
    so an embedding layer can use 0 for padding/unknown and the rest for types.
    """

    def __init__(self) -> None:
        self._type_to_idx: dict[str, int] = {}
        self._frozen = False

    def add_type(self, type_name: str) -> int:
        """Register a node type and return its index. Idempotent for existing types."""
        if type_name in self._type_to_idx:
            return self._type_to_idx[type_name]
        if self._frozen:
            return UNK_TYPE_INDEX
        idx = len(self._type_to_idx) + 1  # 1-based; 0 reserved for UNK
        self._type_to_idx[type_name] = idx
        return idx

    def add_graph_dict(self, graph_dict: dict) -> None:
        """Register all node types present in a graph dict from smt_to_graph()."""
        for node in graph_dict.get("nodes", {}).values():
            if isinstance(node, dict) and "type" in node:
                self.add_type(node["type"])

    def get_index(self, type_name: str) -> int:
        """Return index for type_name; UNK_TYPE_INDEX if unknown."""
        return self._type_to_idx.get(type_name, UNK_TYPE_INDEX)

    def freeze(self) -> None:
        """Stop adding new types; future unknown types will map to UNK."""
        self._frozen = True

    def num_types(self) -> int:
        """Number of known types (excluding UNK). Embedding size can be num_types() + 1."""
        return len(self._type_to_idx)

    def type_names(self) -> list[str]:
        """Ordered list of type names (index i+1 corresponds to type_names()[i])."""
        sorted_pairs = sorted(self._type_to_idx.items(), key=lambda p: p[1])
        return [t for t, _ in sorted_pairs]


def build_vocabulary_from_graph_dicts(graph_dicts: list[dict]) -> NodeVocabulary:
    """
    Build a NodeVocabulary from a list of graph dicts (e.g. from smt_to_graph).
    Call freeze() so the vocabulary can be used for inference on new graphs.
    """
    vocab = NodeVocabulary()
    for g in graph_dicts:
        vocab.add_graph_dict(g)
    vocab.freeze()
    return vocab


def graph_dict_to_gin_data(
    graph_dict: dict,
    vocabulary: NodeVocabulary,
) -> Data:
    """
    Convert a graph dict from smt_to_graph() to a PyG Data (node type indices, undirected edges).
    """
    gin_graph = smt_graph_to_gin(graph_dict)
    type_indices = [vocabulary.get_index(t) for t in gin_graph["node_types"]]
    x = torch.tensor(type_indices, dtype=torch.long).unsqueeze(1)  # (n_nodes, 1)
    edge_list = gin_graph["edges"]
    if not edge_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def build_gin_samples(
    instance_paths: list[str],
    graph_dict_by_path: dict[str, dict],
    multi_perf_data: MultiSolverDataset,
    vocabulary: NodeVocabulary,
) -> list[tuple[Data, torch.Tensor, torch.Tensor]]:
    """
    Build list of (Data, y, mask) for instances that have a graph.
    y and mask have shape (K,). mask[s] = 1.0 if PAR2 is available. Only includes paths in graph_dict_by_path.
    """
    K = multi_perf_data.num_solvers()
    samples: list[tuple[Data, torch.Tensor, torch.Tensor]] = []
    for path in instance_paths:
        graph_dict = graph_dict_by_path.get(path)
        if graph_dict is None:
            continue
        data = graph_dict_to_gin_data(graph_dict, vocabulary)
        y = torch.zeros(K, dtype=torch.float32)
        mask = torch.zeros(K, dtype=torch.float32)
        for s in range(K):
            par2 = multi_perf_data.get_par2(path, s)
            if par2 is not None:
                y[s] = par2
                mask[s] = 1.0
        samples.append((data, y, mask))
    return samples


class GINRegressionDataset(Dataset):
    """Torch Dataset of (Data, y, mask) for GIN multi-head PAR2 regression. Use custom collate to batch."""

    def __init__(
        self,
        samples: list[tuple[Data, torch.Tensor, torch.Tensor]],
    ) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        return self.samples[idx]


class GINMultiHeadEHM(torch.nn.Module):
    """
    GIN-based empirical hardness model: one backbone plus K regression heads.

    Shared backbone produces a graph embedding; each head predicts PAR2 for one
    solver. Output shape (batch_size, K). Selection = argmin over heads.
    """

    def __init__(
        self,
        num_node_types: int,
        num_heads: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
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
        # K heads: each maps hidden_dim -> 1
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_heads)
            ]
        )

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
        out = torch.stack([head(g).squeeze(-1) for head in self.heads], dim=1)
        return out  # (batch_size, num_heads)

    def forward_data(self, data: Batch) -> torch.Tensor:
        """Forward pass from a PyG Batch; returns (batch_size, num_heads)."""
        return self.forward(data.x, data.edge_index, data.batch)


class GINSelector(SolverSelector):
    """
    Algorithm selector using a GIN multi-head regressor: predict PAR2 for each
    solver and return the solver with lowest predicted PAR2. Uses fallback
    solver for instances where graph build fails (timeout/error).
    """

    def __init__(
        self,
        model: GINMultiHeadEHM,
        vocabulary: NodeVocabulary,
        fallback_solver_ids: list[int],
        graph_timeout: int,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.fallback_solver_ids = list(fallback_solver_ids)
        self.graph_timeout = graph_timeout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def algorithm_select(self, instance_path: str | Path) -> int:
        path = Path(instance_path)
        graph_dict = build_smt_graph_dict_timeout(path, self.graph_timeout)
        if graph_dict is None:
            _suppress_z3_destructor_noise()
            return self.fallback_solver_ids[0]
        data = graph_dict_to_gin_data(graph_dict, self.vocabulary)
        batch = Batch.from_data_list([data])
        batch = batch.to(self.device)
        with torch.no_grad():
            pred = self.model.forward_data(batch)  # (1, K)
        pred_np = pred[0].cpu().numpy()
        selected = int(pred_np.argmin())
        _suppress_z3_destructor_noise()
        return selected

    @staticmethod
    def load(load_path: str | Path, device: str | None = None) -> GINSelector:
        """Load from a directory containing config.json, model.pt, vocab.json, failed_paths.txt."""
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

        model = GINMultiHeadEHM(
            num_node_types=config["num_node_types"],
            num_heads=config["num_heads"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=0.0,
        )
        model.load_state_dict(torch.load(load_path / "model.pt", map_location="cpu", weights_only=True))
        fallback_solver_ids = config.get("fallback_solver_ids", config.get("timeout_solver_ids"))
        graph_timeout = config["graph_timeout"]
        return GINSelector(
            model=model,
            vocabulary=vocabulary,
            fallback_solver_ids=fallback_solver_ids,
            graph_timeout=graph_timeout,
            device=device,
        )


def _collate_gin_regression(batch):
    """Collate (Data, y, mask) list into (Batch, y stacked, mask stacked)."""
    data_list = [b[0] for b in batch]
    y_list = [b[1] for b in batch]
    mask_list = [b[2] for b in batch]
    return (
        Batch.from_data_list(data_list),
        torch.stack(y_list),
        torch.stack(mask_list),
    )


def _masked_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE over elements where mask > 0; normalize by sum(mask)."""
    diff = (pred - target) ** 2
    masked = diff * mask
    total = masked.sum()
    count = mask.sum().clamp(min=1e-8)
    return total / count


def train_gin_regression(
    multi_perf_data: MultiSolverDataset,
    save_dir: str | Path,
    *,
    graph_timeout: int = 10,
    jobs: int = 1,
    hidden_dim: int = 64,
    num_layers: int = 3,
    num_epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout: float = 0.1,
    device: str | None = None,
    val_ratio: float = 0.1,
    patience: int = 20,
    val_split_seed: int = 42,
    min_epochs: int = 50,
) -> None:
    """
    Build graphs, vocab, dataset; train GINMultiHeadEHM; save model, vocab, config, failed_paths.
    jobs: number of parallel workers for graph building; 1 = sequential.
    If val_ratio > 0 and patience > 0: split off val_ratio of data for validation, stop when
    validation loss does not improve for patience epochs (after at least min_epochs), and restore best checkpoint.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    instance_paths = list(multi_perf_data.keys())
    K = multi_perf_data.num_solvers()
    logging.info("Building graphs for %d instances (timeout=%ds)...", len(instance_paths), graph_timeout)
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

    samples = build_gin_samples(train_paths, graph_by_path, multi_perf_data, vocab)
    dataset = GINRegressionDataset(samples)
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
            collate_fn=_collate_gin_regression,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_gin_regression,
        )
        logging.info(
            "Early stopping: val_ratio=%.2f, patience=%d, min_epochs=%d -> %d train, %d val",
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
            collate_fn=_collate_gin_regression,
        )
        val_loader = None

    model = GINMultiHeadEHM(
        num_node_types=num_node_types,
        num_heads=K,
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
        for batch_data, y, mask in train_loader:
            batch_data = batch_data.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model.forward_data(batch_data)
            loss = _masked_mse_loss(pred, y, mask)
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
                for batch_data, y, mask in val_loader:
                    batch_data = batch_data.to(device)
                    y = y.to(device)
                    mask = mask.to(device)
                    pred = model.forward_data(batch_data)
                    loss = _masked_mse_loss(pred, y, mask)
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
                    "Epoch %d/%d train loss %.4f val loss %.4f%s",
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
                logging.info("Epoch %d/%d train loss %.4f", epoch + 1, num_epochs, avg_loss)

    if best_state_path.exists():
        model.load_state_dict(torch.load(best_state_path, map_location=device, weights_only=True))
        best_state_path.unlink(missing_ok=True)

    fallback_solver_ids = sorted_fallback_solvers(multi_perf_data, failed_list)
    config = {
        "num_node_types": num_node_types,
        "num_heads": K,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "graph_timeout": graph_timeout,
        "timeout": multi_perf_data.get_timeout(),
        "fallback_solver_ids": fallback_solver_ids,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    torch.save(model.state_dict(), save_dir / "model.pt")
    with open(save_dir / "vocab.json", "w") as f:
        json.dump({"type_names": vocab.type_names()}, f)
    with open(save_dir / "failed_paths.txt", "w") as f:
        for p in failed_list:
            f.write(p + "\n")
    logging.info("Saved model, vocab, config, failed_paths to %s", save_dir)
