from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def load_graph_pickle(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        graph_data = pickle.load(handle)
    if not graph_data:
        raise ValueError(f"No graphs found in {path}")
    return graph_data


def save_graph_pickle(graph_data: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(graph_data, handle)


def infer_feature_dims(graph_data: list[dict[str, Any]]) -> tuple[int, int]:
    first = graph_data[0]
    node_dim = int(first["node_features"].shape[1])
    edge_dim = int(first["edge_features"].shape[1])
    return node_dim, edge_dim

