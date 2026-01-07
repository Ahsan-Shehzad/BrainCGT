from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logging(log_file: Optional[str] = None):
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level="INFO")
    return logger


def mkdirp(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_subjects_csv(subjects_csv: str) -> pd.DataFrame:
    df = pd.read_csv(subjects_csv)
    required = {"subject_id", "fmri_path", "site_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"subjects_csv missing required columns: {sorted(missing)}")

    if "confounds_path" not in df.columns:
        df["confounds_path"] = ""

    df["subject_id"] = df["subject_id"].astype(str)
    df["fmri_path"] = df["fmri_path"].astype(str)
    df["confounds_path"] = df["confounds_path"].astype(str)
    df["site_id"] = df["site_id"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro")),
    }
    try:
        if y_prob.ndim == 1 and len(np.unique(y_true)) == 2:
            out["auc"] = float(roc_auc_score(y_true, y_prob))
        elif y_prob.ndim == 2:
            out["auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        else:
            out["auc"] = float("nan")
    except Exception:
        out["auc"] = float("nan")
    return out


def stratified_kfold(subject_ids: List[str], labels: List[int], k: int, seed: int) -> List[Dict[str, np.ndarray]]:
    subject_ids = np.asarray(subject_ids)
    labels = np.asarray(labels).astype(int)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(subject_ids, labels):
        folds.append({"train": train_idx, "test": test_idx})
    return folds


def standardize_timeseries(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """ROI-wise z-score over time: X [N,T]."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


def pca_first_component_score(X: np.ndarray) -> np.ndarray:
    """
    Compute a scalar per ROI by extracting a dominant temporal mode shared across ROIs.

    Input: X [N,T]
    Output: a [N,1]
    """
    Xz = standardize_timeseries(X)
    # SVD on (N,T): Xz = U S V^T. Use U[:,0]*S[0] as a stable scalar summary per ROI.
    U, S, _ = np.linalg.svd(Xz, full_matrices=False)
    a = (U[:, [0]] * S[0]).astype(np.float32)
    return a


def adjacency_from_edge_index(edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Reconstruct dense adjacency A [N,N] from sparse edge_index/edge_attr."""
    A = torch.zeros((num_nodes, num_nodes), dtype=edge_attr.dtype, device=edge_attr.device)
    A[edge_index[0], edge_index[1]] = edge_attr.view(-1)
    return A


def to_edge_index(A: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert dense adjacency A [N,N] into edge_index [2,E] and edge_attr [E,1] for nonzero entries."""
    A = np.asarray(A)
    src, dst = np.nonzero(A)
    w = A[src, dst].astype(np.float32)
    edge_index = torch.from_numpy(np.vstack([src, dst]).astype(np.int64))
    edge_attr = torch.from_numpy(w.reshape(-1, 1))
    return edge_index, edge_attr
