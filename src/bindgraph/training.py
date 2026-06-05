from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import GraphDataset, collate_single_graph
from .io import infer_feature_dims
from .metrics import kfold_indices, spearman_rho
from .models import OptimizedGNN


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_epoch(
    model: OptimizedGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        node_features = batch["node_features"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_features = batch["edge_features"].to(device)
        num_edges = batch["num_edges"].to(device)
        activity = batch["activity"].to(device)

        optimizer.zero_grad()
        pred = model(node_features, edge_index, edge_features, num_edges)
        loss = criterion(pred, activity)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * activity.size(0)
        n_samples += activity.size(0)

    return total_loss / max(n_samples, 1)


def evaluate(
    model: OptimizedGNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[float], list[float]]:
    model.eval()
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch in loader:
            node_features = batch["node_features"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_features = batch["edge_features"].to(device)
            num_edges = batch["num_edges"].to(device)
            activity = batch["activity"].to(device)

            pred = model(node_features, edge_index, edge_features, num_edges)
            all_preds.extend(pred.cpu().numpy().flatten().tolist())
            all_labels.extend(activity.cpu().numpy().flatten().tolist())

    return spearman_rho(all_labels, all_preds), all_preds, all_labels


def build_model(
    *,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int = 256,
    num_gnn_layers: int = 2,
    dropout: float = 0.3,
) -> OptimizedGNN:
    return OptimizedGNN(
        num_esm2_layers=1,
        esm2_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        dropout=dropout,
        use_scalar_mix=False,
    )


def run_cross_validation(
    graph_data: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    seed: int,
    device: str = "auto",
    n_splits: int = 5,
    max_epochs: int = 200,
    patience: int = 50,
    hidden_dim: int = 256,
    num_gnn_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
) -> list[dict[str, int | float]]:
    set_random_seed(seed)
    torch_device = resolve_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_dim, edge_dim = infer_feature_dims(graph_data)
    fold_results: list[dict[str, int | float]] = []

    with (output_dir / "training.log").open("w") as log_file:
        log_file.write(f"Training start: {datetime.now()}\n")
        log_file.write(f"seed={seed}, device={torch_device}\n")
        log_file.write(
            f"samples={len(graph_data)}, node_dim={node_dim}, edge_dim={edge_dim}\n\n"
        )

        for fold_idx, (train_idx, val_idx) in enumerate(
            kfold_indices(len(graph_data), n_splits, seed), 1
        ):
            train_data = [graph_data[i] for i in train_idx]
            val_data = [graph_data[i] for i in val_idx]

            train_loader = DataLoader(
                GraphDataset(train_data),
                batch_size=1,
                shuffle=True,
                collate_fn=collate_single_graph,
            )
            val_loader = DataLoader(
                GraphDataset(val_data),
                batch_size=1,
                shuffle=False,
                collate_fn=collate_single_graph,
            )

            model = build_model(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                num_gnn_layers=num_gnn_layers,
                dropout=dropout,
            ).to(torch_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.MSELoss()

            best_val_rho = -float("inf")
            best_epoch = 0
            patience_counter = 0

            header = f"Fold {fold_idx}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}"
            print(header, flush=True)
            log_file.write(header + "\n")

            for epoch in range(1, max_epochs + 1):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, torch_device)
                val_rho, _, _ = evaluate(model, val_loader, torch_device)

                if not np.isnan(val_rho) and val_rho > best_val_rho:
                    best_val_rho = val_rho
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(model.state_dict(), output_dir / f"fold{fold_idx}_best.pth")
                else:
                    patience_counter += 1

                if epoch == 1 or epoch % 20 == 0:
                    msg = (
                        f"  epoch={epoch:03d} loss={train_loss:.4f} "
                        f"val_rho={val_rho:.4f} best={best_val_rho:.4f}"
                    )
                    print(msg, flush=True)
                    log_file.write(msg + "\n")

                if patience_counter >= patience:
                    break

            fold_results.append(
                {
                    "fold": fold_idx,
                    "val_rho": best_val_rho,
                    "best_epoch": best_epoch,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                }
            )
            print(f"Fold {fold_idx} done: val_rho={best_val_rho:.4f}", flush=True)

        val_rhos = np.array([row["val_rho"] for row in fold_results], dtype=np.float64)
        mean_rho = float(np.mean(val_rhos))
        sd_rho = float(np.std(val_rhos, ddof=1))
        log_file.write(f"\nMean Spearman rho: {mean_rho:.4f} +/- {sd_rho:.4f}\n")
        log_file.write(f"Training end: {datetime.now()}\n")

    write_fold_results(output_dir / "fold_results.csv", fold_results)
    return fold_results


def write_fold_results(path: str | Path, fold_results: list[dict[str, int | float]]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "val_rho", "best_epoch", "n_train", "n_val"],
        )
        writer.writeheader()
        writer.writerows(fold_results)
