#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from bindgraph.graph import load_single_pair_embedding, make_graph_record
from bindgraph.io import save_graph_pickle


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as handle:
        if path.suffix.lower() == ".json":
            return json.load(handle)
        return yaml.safe_load(handle)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = read_rows(cfg["input_csv"])

    embedding_cfg = cfg.get("embedding", {})
    graph_cfg = cfg.get("graph", {})
    embedding_dir = Path(cfg["embedding_dir"])
    name_column = cfg.get("name_column", "name")
    activity_column = cfg.get("activity_column", "sensing folds")
    top_k = int(graph_cfg.get("top_k", 8))
    filename_template = embedding_cfg.get("filename_template", "{name}.npz")
    lowercase_name = bool(embedding_cfg.get("lowercase_name", False))

    graph_data = []
    failed = []

    row_iter = tqdm(rows, desc="Preparing graphs") if tqdm is not None else rows
    for row in row_iter:
        name = row[name_column]
        file_name_token = name.lower() if lowercase_name else name
        npz_file = embedding_dir / filename_template.format(name=file_name_token)
        try:
            single, pair = load_single_pair_embedding(
                npz_file,
                embedding_format=embedding_cfg.get("format", "custom"),
                single_key=embedding_cfg.get("single_key"),
                pair_key=embedding_cfg.get("pair_key"),
            )
            metadata = {key: value for key, value in row.items() if key not in {activity_column}}
            record = make_graph_record(
                name=name,
                single_embedding=single,
                pair_embedding=pair,
                activity=float(row[activity_column]),
                top_k=top_k,
                metadata=metadata,
            )
            graph_data.append(record)
        except Exception as exc:
            failed.append((name, str(exc)))

    output_pkl = Path(cfg["output_pkl"])
    save_graph_pickle(graph_data, output_pkl)

    print(f"Saved {len(graph_data)} graphs to {output_pkl}")
    if failed:
        failed_file = output_pkl.with_suffix(".failed.tsv")
        with failed_file.open("w") as handle:
            for name, reason in failed:
                handle.write(f"{name}\t{reason}\n")
        print(f"Failed {len(failed)} samples; see {failed_file}")


if __name__ == "__main__":
    main()
