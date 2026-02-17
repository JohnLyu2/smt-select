"""GIN model and data conversion (PAR2 prediction for algorithm selection)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_mean_pool

from .graph_rep import smt_graph_to_gin

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
