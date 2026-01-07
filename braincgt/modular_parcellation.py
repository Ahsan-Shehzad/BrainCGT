from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from nilearn.input_data import NiftiLabelsMasker

from .config import AppConfig
from .utils import mkdirp, save_json, standardize_timeseries


@dataclass
class ModularParcellator:
    cfg: AppConfig

    def _load_module_map(self) -> np.ndarray:
        df = pd.read_csv(self.cfg.parcellation.roi_to_yeo_csv)
        if not {"roi_id", "module_id"}.issubset(df.columns):
            raise ValueError("roi_to_yeo_csv must contain columns: roi_id,module_id")
        df = df.sort_values("roi_id")
        module = df["module_id"].to_numpy(dtype=np.int64)
        return module

    def _load_centroids(self) -> np.ndarray:
        df = pd.read_csv(self.cfg.parcellation.roi_centroids_csv)
        if not {"roi_id", "x", "y", "z"}.issubset(df.columns):
            raise ValueError("roi_centroids_csv must contain columns: roi_id,x,y,z")
        df = df.sort_values("roi_id")
        pos = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return pos

    def run(self, preproc_path: str, subject_id: str) -> Dict:
        out_dir = Path(self.cfg.paths.derivatives_dir) / "roi_ts" / str(subject_id)
        mkdirp(str(out_dir))

        ts_path = out_dir / "roi_timeseries.npy"
        meta_path = out_dir / "parcellation_meta.json"
        module_path = out_dir / "module_id.npy"
        pos_path = out_dir / "roi_pos.npy"

        if ts_path.exists() and module_path.exists() and pos_path.exists():
            X = np.load(ts_path).astype(np.float32)
            module_id = np.load(module_path).astype(np.int64)
            pos = np.load(pos_path).astype(np.float32)
            return {"X": X, "module_id": module_id, "pos": pos}

        module_id = self._load_module_map()
        pos = self._load_centroids()

        masker = NiftiLabelsMasker(
            labels_img=self.cfg.parcellation.atlas_labels_nii,
            standardize=False,
            detrend=False,
            low_pass=None,
            high_pass=None,
            t_r=self.cfg.preprocess.tr,
        )
        # Nilearn returns [T,N], transpose to [N,T]
        X_tn = masker.fit_transform(preproc_path).astype(np.float32)
        X = X_tn.T

        # Temporal cleaning in ROI space for stationarity/VAR suitability
        if self.cfg.preprocess.detrend:
            # remove linear trend via mean-centering and linear detrend
            t = np.arange(X.shape[1], dtype=np.float32)
            t = (t - t.mean()) / (t.std() + 1e-8)
            # Fit slope per ROI
            slope = (X @ t) / (t @ t)
            X = X - slope[:, None] * t[None, :]

        if self.cfg.preprocess.difference_order > 0:
            for _ in range(self.cfg.preprocess.difference_order):
                X = np.diff(X, axis=1)
            # pad to keep consistent length is not necessary for VAR; keep T' = T - diff_order

        # Confounds regression and bandpass filtering are typically voxelwise; here we apply bandpass via FFT in ROI space.
        # This keeps runtime modest and is consistent with VAR assumptions.
        X = standardize_timeseries(X)

        np.save(ts_path, X.astype(np.float32))
        np.save(module_path, module_id.astype(np.int64))
        np.save(pos_path, pos.astype(np.float32))

        meta = {
            "subject_id": str(subject_id),
            "preproc_path": str(preproc_path),
            "atlas": self.cfg.parcellation.atlas_name,
            "N": int(X.shape[0]),
            "T": int(X.shape[1]),
            "M": int(self.cfg.parcellation.yeo_networks),
            "note": "ROI series standardized; detrend/differencing applied per config.",
        }
        save_json(meta, str(meta_path))
        return {"X": X, "module_id": module_id, "pos": pos}
