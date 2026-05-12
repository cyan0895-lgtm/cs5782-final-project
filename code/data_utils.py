"""
data_utils.py — load raw Planetoid dataset files into a PyG Data object.

Expected directory layout:
    data/
    ├── cora/      → ind.cora.*
    ├── citeseer/  → ind.citeseer.*
    └── pubmed/    → ind.pubmed.*
"""

import os
import pickle
import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.data import Data


SPLIT = {
    "cora":     {"train": 140, "val": 500, "test": 1000},
    "citeseer": {"train": 120, "val": 500, "test": 1000},
    "pubmed":   {"train":  60, "val": 500, "test": 1000},
}


def _load_file(path: str):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _to_tensor(mat) -> torch.Tensor:
    if sp.issparse(mat):
        mat = mat.tocsr().astype(np.float32)
        return torch.from_numpy(mat.toarray())
    return torch.tensor(np.array(mat), dtype=torch.float32)


class DatasetInfo:
    def __init__(self, num_features: int, num_classes: int, name: str):
        self.num_features = num_features
        self.num_classes  = num_classes
        self.name         = name


def load_dataset(name: str, data_root: str = "data"):
    """
    Load a Planetoid dataset from raw files.

    Key fix: tx[i] corresponds to test_idx[i] (original unsorted order),
    NOT test_idx_sorted[i]. Citeseer's tx has 1015 rows but test_idx only
    has 1000 entries — the extra 15 rows are isolated nodes that we ignore.
    """
    name_lower = name.lower()
    if name_lower not in SPLIT:
        raise ValueError(f"Unknown dataset '{name}'. Choose: {list(SPLIT)}")

    folder = os.path.join(data_root, name_lower)
    p = os.path.join(folder, f"ind.{name_lower}")

    # ── Load raw files ────────────────────────────────────────────────────────
    allx  = _load_file(f"{p}.allx")
    tx    = _load_file(f"{p}.tx")
    ally  = _load_file(f"{p}.ally")
    ty    = _load_file(f"{p}.ty")
    graph = _load_file(f"{p}.graph")
    test_idx = np.loadtxt(f"{p}.test.index", dtype=np.int64)

    num_nodes = max(graph.keys()) + 1
    feat_allx = _to_tensor(allx)
    feat_tx   = _to_tensor(tx)
    F_dim     = feat_allx.size(1)

    # ── Feature matrix ────────────────────────────────────────────────────────
    features = torch.zeros(num_nodes, F_dim)
    features[:feat_allx.size(0)] = feat_allx

    # tx[i] → test_idx[i] (original order, not sorted).
    # For Citeseer, tx has 1015 rows but test_idx has 1000 — only use first
    # len(test_idx) rows of tx, ignoring the extra 15 isolated-node rows.
    for i, idx in enumerate(test_idx):
        features[idx] = feat_tx[i]

    # ── Label matrix ──────────────────────────────────────────────────────────
    labels_allx = _to_tensor(ally)
    labels_ty   = _to_tensor(ty)
    C = labels_allx.size(1)

    labels = torch.zeros(num_nodes, C)
    labels[:labels_allx.size(0)] = labels_allx

    for i, idx in enumerate(test_idx):
        labels[idx] = labels_ty[i]

    y_int = labels.argmax(dim=1)

    # ── Edge index ────────────────────────────────────────────────────────────
    rows, cols = [], []
    for src, neighbors in graph.items():
        for dst in neighbors:
            rows.append(src); cols.append(dst)
            rows.append(dst); cols.append(src)
    edge_index = torch.unique(
        torch.tensor([rows, cols], dtype=torch.long), dim=1
    )

    # ── Masks ─────────────────────────────────────────────────────────────────
    split   = SPLIT[name_lower]
    n_train = split["train"]
    n_val   = split["val"]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:n_train]               = True
    val_mask[n_train:n_train + n_val]  = True
    test_mask[test_idx]                = True

    data = Data(
        x=features, edge_index=edge_index, y=y_int,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )

    dataset_info = DatasetInfo(
        num_features=F_dim,
        num_classes=int(y_int.max().item()) + 1,
        name=name,
    )

    print(f"[{name}] nodes={num_nodes} | edges={edge_index.size(1)} | "
          f"features={F_dim} | classes={dataset_info.num_classes} | "
          f"train/val/test={n_train}/{n_val}/{split['test']}")

    return dataset_info, data
