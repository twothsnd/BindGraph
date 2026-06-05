from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .graph import construct_topk_graph
from .training import build_model, resolve_device


def load_ensemble(
    model_dir: str | Path,
    *,
    node_dim: int = 384,
    edge_dim: int = 128,
    hidden_dim: int = 256,
    num_gnn_layers: int = 2,
    dropout: float = 0.3,
    n_folds: int = 5,
    device: str = "auto",
) -> tuple[list[torch.nn.Module], torch.device]:
    torch_device = resolve_device(device)
    model_dir = Path(model_dir)
    models = []

    for fold in range(1, n_folds + 1):
        model = build_model(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
        ).to(torch_device)
        state_dict = torch.load(
            model_dir / f"fold{fold}_best.pth",
            map_location=torch_device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    return models, torch_device


def predict_with_ensemble(
    models: list[torch.nn.Module],
    device: torch.device,
    single_embedding: np.ndarray,
    pair_embedding: np.ndarray,
    *,
    top_k: int = 8,
) -> tuple[float, float, list[float]]:
    edge_index, edge_features = construct_topk_graph(pair_embedding, k=top_k)

    node_t = torch.as_tensor(single_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long).unsqueeze(0).to(device)
    edge_features_t = torch.as_tensor(edge_features, dtype=torch.float32).unsqueeze(0).to(device)
    num_edges_t = torch.as_tensor([edge_index.shape[1]], dtype=torch.long).to(device)

    fold_preds = []
    with torch.no_grad():
        for model in models:
            pred = model(node_t, edge_index_t, edge_features_t, num_edges_t)
            fold_preds.append(float(pred.cpu().item()))

    return float(np.mean(fold_preds)), float(np.std(fold_preds)), fold_preds

