from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


SEQUENCE_OFFSETS = (-2, -1, 1, 2)


def load_single_pair_embedding(
    npz_file: str | Path,
    *,
    embedding_format: str,
    single_key: str | None = None,
    pair_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load single and pair embeddings from an `.npz` file."""
    data = np.load(npz_file)

    if embedding_format == "boltz2":
        single = data["s"][0]
        pair = data["z"][0]
    elif embedding_format == "af3_monomer":
        single = data["single_embeddings"]
        pair = data["pair_embeddings"]
    elif embedding_format == "custom":
        if not single_key or not pair_key:
            raise ValueError("custom embedding format requires single_key and pair_key")
        single = data[single_key]
        pair = data[pair_key]
    else:
        raise ValueError(f"Unsupported embedding format: {embedding_format}")

    single = np.asarray(single, dtype=np.float32)
    pair = np.asarray(pair, dtype=np.float32)
    validate_embedding_shapes(single, pair)
    return single, pair


def validate_embedding_shapes(single: np.ndarray, pair: np.ndarray) -> None:
    if single.ndim != 2:
        raise ValueError(f"single embedding must be 2D, got shape {single.shape}")
    if pair.ndim != 3:
        raise ValueError(f"pair embedding must be 3D, got shape {pair.shape}")
    if single.shape[0] != pair.shape[0] or pair.shape[0] != pair.shape[1]:
        raise ValueError(f"shape mismatch: single={single.shape}, pair={pair.shape}")


def construct_topk_graph(
    pair_embeddings: np.ndarray,
    *,
    k: int = 8,
    sequence_offsets: tuple[int, ...] = SEQUENCE_OFFSETS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a sparse directed graph from pair embeddings.

    Each node connects to sequence-neighbor nodes and to the top-k non-local
    nodes ranked by pair-vector L2 norm.
    """
    if pair_embeddings.ndim != 3:
        raise ValueError(f"pair_embeddings must be 3D, got shape {pair_embeddings.shape}")

    seq_len = pair_embeddings.shape[0]
    edge_dim = pair_embeddings.shape[-1]
    pair_norms = np.linalg.norm(pair_embeddings, axis=-1)

    edge_index_list: list[list[int]] = []
    edge_feature_list: list[np.ndarray] = []

    for i in range(seq_len):
        sequence_neighbors = []
        for offset in sequence_offsets:
            j = i + offset
            if 0 <= j < seq_len:
                sequence_neighbors.append(j)
                edge_index_list.append([i, j])
                edge_feature_list.append(pair_embeddings[i, j])

        scores = pair_norms[i].copy()
        scores[i] = -np.inf
        for j in sequence_neighbors:
            scores[j] = -np.inf

        if k > 0:
            top_k_indices = np.argsort(scores)[-k:]
            for j in top_k_indices:
                if scores[j] > -np.inf:
                    edge_index_list.append([i, int(j)])
                    edge_feature_list.append(pair_embeddings[i, j])

    if not edge_index_list:
        edge_index = np.array([[0], [0]], dtype=np.int64)
        edge_features = np.zeros((1, edge_dim), dtype=pair_embeddings.dtype)
        return edge_index, edge_features

    edge_index = np.asarray(edge_index_list, dtype=np.int64).T
    edge_features = np.asarray(edge_feature_list, dtype=np.float32)
    return edge_index, edge_features


def make_graph_record(
    *,
    name: str,
    single_embedding: np.ndarray,
    pair_embedding: np.ndarray,
    activity: float | None = None,
    top_k: int = 8,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    edge_index, edge_features = construct_topk_graph(pair_embedding, k=top_k)
    record: dict[str, Any] = {
        "name": name,
        "node_features": np.asarray(single_embedding, dtype=np.float32),
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": int(single_embedding.shape[0]),
        "num_edges": int(edge_index.shape[1]),
    }
    if activity is not None:
        record["activity"] = float(activity)
    if metadata:
        record.update(metadata)
    return record

