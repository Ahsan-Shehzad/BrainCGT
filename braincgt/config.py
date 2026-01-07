from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class PathsConfig(BaseModel):
    raw_dir: str = Field(..., description="Root directory containing raw data.")
    derivatives_dir: str = Field(..., description="Root directory for derived artifacts.")
    atlas_dir: str = Field(..., description="Directory containing atlas resources.")
    splits_dir: str = Field(..., description="Directory for dataset splits.")
    results_dir: str = Field(..., description="Directory for run outputs (checkpoints, metrics, explanations).")


class PreprocessConfig(BaseModel):
    tr: float = Field(2.0, gt=0.0, description="Repetition time in seconds.")
    bandpass_low: float = Field(0.01, ge=0.0)
    bandpass_high: float = Field(0.1, gt=0.0)
    smoothing_fwhm_mm: float = Field(6.0, ge=0.0)
    detrend: bool = True
    difference_order: int = Field(1, ge=0, le=2, description="Temporal differencing order.")
    regress_confounds: bool = True
    confounds_columns: List[str] = Field(default_factory=list)
    combat: bool = False

    @model_validator(mode="after")
    def _validate_bandpass(self) -> "PreprocessConfig":
        if self.bandpass_low >= self.bandpass_high:
            raise ValueError("bandpass_low must be strictly smaller than bandpass_high.")
        if self.bandpass_high > 0.5 / self.tr:
            raise ValueError("bandpass_high exceeds Nyquist frequency (0.5/TR).")
        return self


class ParcellationConfig(BaseModel):
    atlas_name: str = "brainnetome_246"
    yeo_networks: int = Field(7, ge=2, le=17)
    atlas_labels_nii: str = Field(..., description="Path to atlas labels NIfTI (integer ROI labels).")
    roi_metadata_csv: str = Field(..., description="CSV with ROI metadata (roi_id, name, ...).")
    roi_to_yeo_csv: str = Field(..., description="CSV mapping roi_id to module_id in [0..M-1].")
    roi_centroids_csv: str = Field(..., description="CSV with ROI centroids (roi_id,x,y,z) in MNI.")


class GrangerConfig(BaseModel):
    max_lag_p: int = Field(5, ge=1, le=20)
    bic_select: bool = True
    fdr_alpha: float = Field(0.05, gt=0.0, lt=1.0)
    standardize: bool = True
    per_module_pair: bool = True


class ModelConfig(BaseModel):
    hidden_dim: int = Field(128, ge=16, le=1024)
    num_layers: int = Field(3, ge=1, le=24)
    num_heads: int = Field(4, ge=1, le=16)
    attn_dim: int = Field(32, ge=8, le=256)
    dropout: float = Field(0.2, ge=0.0, lt=1.0)
    gamma_causal_bias: float = Field(1.0, ge=0.0, description="Causal bias strength added to attention logits.")
    use_posenc: bool = True
    posenc_dim: int = Field(24, ge=0, le=256)
    num_classes: int = Field(2, ge=2, le=32)

    @model_validator(mode="after")
    def _validate_heads(self) -> "ModelConfig":
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        return self


class TrainConfig(BaseModel):
    seed: int = Field(42, ge=0, le=2**31 - 1)
    epochs: int = Field(200, ge=1, le=5000)
    batch_size: int = Field(16, ge=1, le=256)
    lr: float = Field(2e-4, gt=0.0)
    weight_decay: float = Field(1e-4, ge=0.0)
    early_stop_patience: int = Field(20, ge=1, le=500)
    grad_clip: float = Field(1.0, ge=0.0)
    folds: int = Field(5, ge=2, le=20)
    num_workers: int = Field(2, ge=0, le=32)
    amp: bool = False
    monitor_metric: str = Field("auc", description="Early stopping metric: 'auc' or 'f1'.")
    checkpoint_every: int = Field(1, ge=1, le=100)

    @field_validator("monitor_metric")
    @classmethod
    def _validate_monitor_metric(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"auc", "f1"}:
            raise ValueError("monitor_metric must be 'auc' or 'f1'.")
        return v


class ExplainConfig(BaseModel):
    method: str = Field("gnnexplainer")
    epochs: int = Field(200, ge=1, le=2000)
    lr: float = Field(0.01, gt=0.0)
    edge_sparsity_lambda: float = Field(0.005, ge=0.0)
    topk_edges: int = Field(200, ge=1)
    topk_nodes: int = Field(50, ge=1)


class AppConfig(BaseModel):
    paths: PathsConfig
    preprocess: PreprocessConfig
    parcellation: ParcellationConfig
    granger: GrangerConfig
    model: ModelConfig
    train: TrainConfig
    explain: ExplainConfig


def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    """
    Parse CLI overrides of the form:
      paths.raw_dir=data/raw
      model.hidden_dim=256
    """
    out: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value.")
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()

        # type coercion
        if val.lower() in {"true", "false"}:
            v_parsed: Any = val.lower() == "true"
        else:
            try:
                if "." in val:
                    f = float(val)
                    v_parsed = int(f) if f.is_integer() else f
                else:
                    v_parsed = int(val)
            except ValueError:
                v_parsed = val

        cursor = out
        parts = key.split(".")
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = v_parsed
    return out


def get_cfg(yaml_path: str, overrides: Optional[List[str]] = None) -> AppConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    if overrides:
        base_cfg = _deep_update(base_cfg, parse_overrides(overrides))

    cfg = AppConfig.model_validate(base_cfg)

    # Normalize key paths
    for attr in ["raw_dir", "derivatives_dir", "atlas_dir", "splits_dir", "results_dir"]:
        p = getattr(cfg.paths, attr)
        setattr(cfg.paths, attr, str(Path(p)))

    return cfg
