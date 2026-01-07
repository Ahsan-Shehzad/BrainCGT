from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from braincgt.config import get_cfg
from braincgt.causal_brain_graph_transformer import BrainCGT
from braincgt.causal_interpretability import CausalExplainer
from braincgt.utils import init_logging, mkdirp, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--graph_pt", required=True, type=str, help="Path to a single graph .pt file")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--target", type=int, default=None)
    args = ap.parse_args()

    cfg = get_cfg(args.config, overrides=args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(args.graph_pt)
    data = data.to(device)

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

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    out_dir = args.out_dir or str(Path(cfg.paths.results_dir) / "explanations")
    mkdirp(out_dir)
    init_logging(str(Path(out_dir) / "explain.log"))

    explainer = CausalExplainer(cfg)
    explanation = explainer.explain(model, data, target=args.target)
    subject_id = getattr(data, "subject_id", Path(args.graph_pt).stem)
    explainer.save(out_dir, subject_id, explanation, data.edge_index)

    logger.info(f"Saved explanation for {subject_id} to {Path(out_dir) / str(subject_id)}")


if __name__ == "__main__":
    main()
