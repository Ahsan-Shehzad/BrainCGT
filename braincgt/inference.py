from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from loguru import logger

from .config import get_cfg, AppConfig
from .causal_brain_graph_transformer import BrainCGT
from .causal_brain_network_construction import GraphFolderDataset
from .utils import init_logging, mkdirp, save_json


def load_model_from_checkpoint(cfg: AppConfig, checkpoint_path: str, device: torch.device) -> BrainCGT:
    model = BrainCGT(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        attn_dim=cfg.model.attn_dim,
        dropout=cfg.model.dropout,
        gamma=cfg.model.gamma_causal_bias,
        use_posenc=cfg.model.use_posenc,
        posenc_dim=cfg.model.posenc_dim,
        num_classes=cfg.model.num_classes,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict(model: BrainCGT, graphs: List, batch_size: int, device: torch.device) -> pd.DataFrame:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        prob = out["prob"].detach().cpu().numpy()
        pred = prob.argmax(axis=1)
        y = batch.y.view(-1).detach().cpu().numpy().astype(int) if hasattr(batch, "y") and batch.y is not None else None

        # subject_id and site_id are concatenated by DataLoader into python lists (PyG preserves them)
        subj_ids = getattr(batch, "subject_id", ["NA"] * prob.shape[0])
        site_ids = getattr(batch, "site_id", ["NA"] * prob.shape[0])
        if isinstance(subj_ids, str):
            subj_ids = [subj_ids]
        if isinstance(site_ids, str):
            site_ids = [site_ids]

        for i in range(prob.shape[0]):
            row = {
                "subject_id": str(subj_ids[i]) if i < len(subj_ids) else "NA",
                "site_id": str(site_ids[i]) if i < len(site_ids) else "NA",
                "pred": int(pred[i]),
            }
            for c in range(prob.shape[1]):
                row[f"prob_{c}"] = float(prob[i, c])
            if y is not None:
                row["label"] = int(y[i])
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--graphs_dir", required=True, type=str)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    cfg = get_cfg(args.config, overrides=args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_csv = str(Path(args.graphs_dir) / "graphs_index.csv")
    dataset = GraphFolderDataset(index_csv)
    graphs = [dataset[i] for i in range(len(dataset))]

    model = load_model_from_checkpoint(cfg, args.checkpoint, device)
    df = predict(model, graphs, args.batch_size, device)

    out_csv = args.out_csv or str(Path(args.graphs_dir) / "predictions.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote predictions to {out_csv}")


if __name__ == "__main__":
    main()
