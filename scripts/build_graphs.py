from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from braincgt.config import get_cfg
from braincgt.causal_brain_network_construction import BrainNetworkBuilder
from braincgt.utils import init_logging, load_subjects_csv, mkdirp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--subjects_csv", required=True, type=str)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--graphs_dir", type=str, default="")
    args = ap.parse_args()

    cfg = get_cfg(args.config, overrides=args.override)
    graphs_dir = args.graphs_dir or str(Path(cfg.paths.derivatives_dir) / "graphs")
    mkdirp(graphs_dir)
    init_logging(str(Path(graphs_dir) / "build_graphs.log"))

    df = load_subjects_csv(args.subjects_csv)
    builder = BrainNetworkBuilder(cfg)
    index_df = builder.build_all(df, graphs_dir)
    logger.info(f"Graphs built. Index saved at {Path(graphs_dir) / 'graphs_index.csv'}")


if __name__ == "__main__":
    main()
