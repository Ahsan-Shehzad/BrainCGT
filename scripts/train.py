from __future__ import annotations

import argparse
from pathlib import Path

from braincgt.config import get_cfg
from braincgt.training import train_kfold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--graphs_dir", required=True, type=str)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--run_tag", type=str, default="braincgt")
    args = ap.parse_args()

    cfg = get_cfg(args.config, overrides=args.override)
    graphs_index_csv = str(Path(args.graphs_dir) / "graphs_index.csv")
    train_kfold(cfg, graphs_index_csv, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
