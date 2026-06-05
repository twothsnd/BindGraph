from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, graph_data: list[dict[str, Any]]):
        self.graph_data = graph_data

    def __len__(self) -> int:
        return len(self.graph_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        graph = self.graph_data[idx]
        return tensorize_graph(graph)


def tensorize_graph(graph: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "node_features": torch.as_tensor(graph["node_features"], dtype=torch.float32),
        "edge_index": torch.as_tensor(graph["edge_index"], dtype=torch.long),
        "edge_features": torch.as_tensor(graph["edge_features"], dtype=torch.float32),
        "num_edges": torch.as_tensor([graph["num_edges"]], dtype=torch.long),
        "activity": torch.as_tensor([graph["activity"]], dtype=torch.float32),
    }


def collate_single_graph(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("This implementation expects batch_size=1 for variable-size graphs")
    item = batch[0]
    return {
        "node_features": item["node_features"].unsqueeze(0),
        "edge_index": item["edge_index"].unsqueeze(0),
        "edge_features": item["edge_features"].unsqueeze(0),
        "num_edges": item["num_edges"],
        "activity": item["activity"].unsqueeze(0),
    }

