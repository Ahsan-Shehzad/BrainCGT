from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import nibabel as nib
from loguru import logger

from .config import AppConfig
from .utils import mkdirp, save_json


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def ensure_fsl_available() -> None:
    if _which("mcflirt") is None or _which("flirt") is None:
        raise RuntimeError(
            "FSL binaries not found on PATH. Install FSL and ensure mcflirt/flirt/fnirt are available."
        )
    if os.environ.get("FSLDIR", "") == "":
        raise RuntimeError("Environment variable FSLDIR is not set. Set FSLDIR to your FSL installation path.")


def _load_confounds(confounds_path: str, columns: list[str]) -> Optional[np.ndarray]:
    if not confounds_path:
        return None
    p = Path(confounds_path)
    if not p.exists():
        return None

    # Accept TSV/CSV
    if p.suffix.lower() == ".tsv":
        df = pd.read_csv(p, sep="\t")
    else:
        df = pd.read_csv(p)

    cols = [c for c in columns if c in df.columns]
    if not cols:
        return None
    return df[cols].to_numpy(dtype=np.float32)


def _compute_fd_from_mcflirt_par(par_path: str) -> float:
    """
    Compute mean framewise displacement (FD) from MCFLIRT motion parameters.
    MCFLIRT par file is 6 columns: rotations (rad) and translations (mm), depending on settings.
    We approximate FD by sum of absolute derivatives with rotation radius 50 mm.
    """
    try:
        mp = np.loadtxt(par_path, dtype=np.float32)  # [T,6]
        if mp.ndim != 2 or mp.shape[1] < 6:
            return float("nan")
        d = np.diff(mp, axis=0)
        rot = d[:, :3] * 50.0
        trans = d[:, 3:6]
        fd = np.abs(rot).sum(axis=1) + np.abs(trans).sum(axis=1)
        return float(np.nanmean(fd))
    except Exception:
        return float("nan")


@dataclass
class FMRI_Preprocessor:
    cfg: AppConfig

    def run_subject(self, subj: Dict) -> Dict:
        """
        Preprocess a single fMRI scan using Nipype+FSL for key steps:
        - Slice timing correction
        - Motion correction (MCFLIRT)
        - Normalization to MNI152 (FLIRT + FNIRT)
        - Spatial smoothing (FSL IsotropicSmooth)

        Temporal denoising (confound regression, bandpass filtering, detrending) is applied in ROI space
        within Modular_Parcellation to avoid huge voxelwise materialization.

        Returns an updated subject dict including:
          - preproc_path
          - motion_fd_mean
          - report_path
        """
        ensure_fsl_available()

        subject_id = str(subj["subject_id"])
        fmri_path = str(subj["fmri_path"])

        out_dir = Path(self.cfg.paths.derivatives_dir) / "preproc" / subject_id
        mkdirp(str(out_dir))
        preproc_path = out_dir / "fmri_preproc.nii.gz"
        report_path = out_dir / "preproc_report.json"

        if preproc_path.exists():
            subj = dict(subj)
            subj["preproc_path"] = str(preproc_path)
            subj["preproc_report"] = str(report_path)
            return subj

        from nipype.interfaces import fsl
        from nipype import Node, Workflow

        # Standard template
        fsldir = os.environ["FSLDIR"]
        mni = Path(fsldir) / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz"
        if not mni.exists():
            raise RuntimeError(f"Cannot find MNI152 template at {mni}")

        work_dir = out_dir / "nipype_work"
        mkdirp(str(work_dir))

        # Nodes
        slicetimer = Node(fsl.SliceTimer(time_repetition=self.cfg.preprocess.tr, interleaved=True), name="slicetimer")
        slicetimer.inputs.in_file = fmri_path

        mcflirt = Node(fsl.MCFLIRT(save_plots=True, mean_vol=True), name="mcflirt")
        flirt = Node(fsl.FLIRT(reference=str(mni), dof=12), name="flirt")
        fnirt = Node(fsl.FNIRT(ref_file=str(mni)), name="fnirt")
        applywarp = Node(fsl.ApplyWarp(ref_file=str(mni)), name="applywarp")

        smooth = Node(fsl.IsotropicSmooth(fwhm=self.cfg.preprocess.smoothing_fwhm_mm), name="smooth")

        wf = Workflow(name=f"preproc_{subject_id}", base_dir=str(work_dir))
        wf.connect([
            (slicetimer, mcflirt, [("slice_time_corrected_file", "in_file")]),
            (mcflirt, flirt, [("out_file", "in_file")]),
            (flirt, fnirt, [("out_matrix_file", "affine_file"),
                            ("out_file", "in_file")]),
            (fnirt, applywarp, [("fieldcoeff_file", "field_file")]),
            (mcflirt, applywarp, [("out_file", "in_file")]),
            (applywarp, smooth, [("out_file", "in_file")]),
        ])

        res = wf.run()

        # Locate output (nipype writes to node output directories)
        # We'll take the smooth output file
        smooth_out = smooth.result.outputs.out_file  # type: ignore[attr-defined]
        shutil.copyfile(smooth_out, preproc_path)

        # Motion par file from MCFLIRT
        par_path = getattr(mcflirt.result.outputs, "par_file", None)  # type: ignore[attr-defined]
        fd_mean = _compute_fd_from_mcflirt_par(par_path) if par_path else float("nan")

        report = {
            "subject_id": subject_id,
            "fmri_path": fmri_path,
            "preproc_path": str(preproc_path),
            "motion_fd_mean": fd_mean,
            "tr": self.cfg.preprocess.tr,
            "smoothing_fwhm_mm": self.cfg.preprocess.smoothing_fwhm_mm,
            "note": "Temporal cleaning (confounds regression, bandpass, detrend, differencing) is applied in ROI space.",
        }
        save_json(report, str(report_path))

        subj = dict(subj)
        subj["preproc_path"] = str(preproc_path)
        subj["preproc_report"] = str(report_path)
        subj["motion_fd_mean"] = fd_mean
        return subj
