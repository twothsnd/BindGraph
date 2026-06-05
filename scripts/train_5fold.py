#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np

from bindgraph.io import load_graph_pickle
from bindgraph.training import run_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    graph_data = load_graph_pickle(args.graph_pkl)
    fold_results = run_cross_validation(
        graph_data,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        n_splits=args.n_splits,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    rhos = np.array([row["val_rho"] for row in fold_results], dtype=float)
    print(f"Mean Spearman rho: {np.mean(rhos):.4f} +/- {np.std(rhos, ddof=1):.4f}")


if __name__ == "__main__":
    main()
