#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from bindgraph.graph import load_single_pair_embedding
from bindgraph.prediction import load_ensemble, predict_with_ensemble


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--embedding-format", choices=["boltz2", "af3_monomer", "custom"], required=True)
    parser.add_argument("--filename-template", required=True)
    parser.add_argument("--name-column", default="mutation_id")
    parser.add_argument("--single-key")
    parser.add_argument("--pair-key")
    parser.add_argument("--lowercase-name", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--node-dim", type=int, default=384)
    parser.add_argument("--edge-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-n", type=int)
    args = parser.parse_args()

    models, device = load_ensemble(
        args.model_dir,
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        n_folds=args.n_folds,
        device=args.device,
    )

    rows = read_rows(args.candidate_csv)
    embedding_dir = Path(args.embedding_dir)
    predictions = []
    failed = []

    row_iter = tqdm(rows, desc="Predicting") if tqdm is not None else rows
    for row in row_iter:
        name = row[args.name_column]
        file_name_token = name.lower() if args.lowercase_name else name
        npz_file = embedding_dir / args.filename_template.format(name=file_name_token)
        try:
            single, pair = load_single_pair_embedding(
                npz_file,
                embedding_format=args.embedding_format,
                single_key=args.single_key,
                pair_key=args.pair_key,
            )
            mean_pred, std_pred, _ = predict_with_ensemble(
                models,
                device,
                single,
                pair,
                top_k=args.top_k,
            )
            out = {
                "name": name,
                "predicted_sensing_fold": mean_pred,
                "prediction_std": std_pred,
            }
            for key, value in row.items():
                if key not in out:
                    out[key] = value
            predictions.append(out)
        except Exception as exc:
            failed.append((name, str(exc)))

    predictions.sort(key=lambda item: float(item["predicted_sensing_fold"]), reverse=True)
    total = len(predictions)
    for idx, row in enumerate(predictions, 1):
        row["rank"] = idx
        row["percentile"] = idx / total * 100 if total else 0

    if args.top_n:
        predictions = predictions[: args.top_n]

    write_rows(args.output_csv, predictions)
    print(f"Saved {len(predictions)} predictions to {args.output_csv}")

    if failed:
        failed_file = Path(args.output_csv).with_suffix(".failed.tsv")
        with failed_file.open("w") as handle:
            for name, reason in failed:
                handle.write(f"{name}\t{reason}\n")
        print(f"Failed {len(failed)} candidates; see {failed_file}")


if __name__ == "__main__":
    main()
