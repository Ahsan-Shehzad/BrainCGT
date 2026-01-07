from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from braincgt.config import get_cfg
from braincgt.fmri_preprocessing import FMRI_Preprocessor
from braincgt.utils import init_logging, load_subjects_csv, mkdirp, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--subjects_csv", required=True, type=str)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = get_cfg(args.config, overrides=args.override)

    run_dir = Path(cfg.paths.derivatives_dir) / "logs"
    mkdirp(str(run_dir))
    init_logging(str(run_dir / "preprocess.log"))

    df = load_subjects_csv(args.subjects_csv)
    preproc = FMRI_Preprocessor(cfg)

    out_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        subj = row.to_dict()
        try:
            out = preproc.run_subject(subj)
            out_rows.append(out)
        except Exception as e:
            logger.error(f"Preprocess failed for {subj['subject_id']}: {e}")

    out_df = pd.DataFrame(out_rows)
    out_path = Path(cfg.paths.derivatives_dir) / "preproc_index.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(f"Wrote preprocessing index to {out_path}")


if __name__ == "__main__":
    main()
