from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConditionedGraphConv(nn.Module):
    """Edge-conditioned graph convolution with per-target attention."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim

        self.W_self = nn.Linear(node_dim, out_dim)
        self.W_msg = nn.Linear(node_dim + edge_dim, out_dim)
        self.attention = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        num_edges_per_sample: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = node_features.shape
        self_out = self.W_self(node_features)
        aggregated = torch.zeros(
            batch_size, seq_len, self.out_dim, device=node_features.device
        )

        for b in range(batch_size):
            if num_edges_per_sample is not None:
                actual_num_edges = int(num_edges_per_sample[b].item())
                src_nodes = edge_index[b, 0, :actual_num_edges]
                tgt_nodes = edge_index[b, 1, :actual_num_edges]
                edges = edge_features[b, :actual_num_edges]
            else:
                src_nodes = edge_index[b, 0]
                tgt_nodes = edge_index[b, 1]
                edges = edge_features[b]

            if len(src_nodes) == 0:
                continue

            h_src = node_features[b, src_nodes]
            h_tgt = node_features[b, tgt_nodes]

            attn_input = torch.cat([h_tgt, h_src, edges], dim=-1)
            attn_scores = self.attention(attn_input).squeeze(-1)

            attn_weights = torch.zeros_like(attn_scores)
            for node_idx in range(seq_len):
                mask = tgt_nodes == node_idx
                if mask.sum() > 0:
                    attn_weights[mask] = F.softmax(attn_scores[mask], dim=0)

            messages = self.W_msg(torch.cat([h_src, edges], dim=-1))
            weighted_messages = messages * attn_weights.unsqueeze(-1)
            aggregated[b].index_add_(0, tgt_nodes, weighted_messages)

        updated = self_out + aggregated
        updated = self.dropout(updated)
        updated = self.layer_norm(updated)
        return F.relu(updated)


class OptimizedGNN(nn.Module):
    """BindGraph sparse pair-embedding GNN.

    The constructor keeps legacy argument names so checkpoints from the local
    training scripts can be loaded without renaming state-dict keys.
    """

    def __init__(
        self,
        num_esm2_layers: int = 1,
        esm2_dim: int = 384,
        edge_dim: int = 128,
        hidden_dim: int = 256,
        num_gnn_layers: int = 2,
        dropout: float = 0.3,
        use_scalar_mix: bool = False,
    ):
        super().__init__()
        if use_scalar_mix:
            raise ValueError("BindGraph expects use_scalar_mix=False")
        _ = num_esm2_layers

        self.use_scalar_mix = False
        self.node_proj = nn.Linear(esm2_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [
                EdgeConditionedGraphConv(hidden_dim, hidden_dim, hidden_dim, dropout)
                for _ in range(num_gnn_layers)
            ]
        )
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        num_edges_per_sample: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_h = self.node_proj(node_features)
        edge_h = self.edge_proj(edge_features)

        for gnn_layer in self.gnn_layers:
            node_h = gnn_layer(node_h, edge_index, edge_h, num_edges_per_sample)

        attn_scores = self.pool_attn(node_h).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        graph_emb = (node_h * attn_weights.unsqueeze(-1)).sum(dim=1)
        return self.mlp(graph_emb)
