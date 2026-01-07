from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data

from .config import AppConfig
from .utils import mkdirp, save_json


@dataclass
class CausalExplainer:
    cfg: AppConfig

    def explain(self, model: torch.nn.Module, data: Data, target: Optional[int] = None) -> Dict:
        """
        Directed edge explanation using PyG's GNNExplainer-style interface.
        Returns edge_mask aligned with data.edge_index (directed).
        """
        try:
            from torch_geometric.explain import Explainer, GNNExplainer
            from torch_geometric.explain.config import ModelConfig, MaskType, ModelMode
        except Exception as e:
            raise RuntimeError("torch_geometric.explain is required for explanations. Install torch-geometric>=2.3.") from e

        model.eval()

        # Build an explainer that expects (x, edge_index, ...) style forward.
        # Our model takes a Data object; we therefore provide a wrapper forward.
        class _Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x, edge_index, edge_attr, pos, module_id, batch):
                tmp = Data()
                tmp.a = x
                tmp.edge_index = edge_index
                tmp.edge_attr = edge_attr
                tmp.pos = pos
                tmp.module_id = module_id
                tmp.batch = batch
                out = self.m(tmp)
                return out["logits"]

        wrapper = _Wrapper(model)

        explainer = Explainer(
            model=wrapper,
            algorithm=GNNExplainer(epochs=int(self.cfg.explain.epochs), lr=float(self.cfg.explain.lr)),
            explanation_type="model",
            model_config=ModelConfig(
                mode=ModelMode.classification,
                task_level="graph",
                return_type="log_probs",
            ),
            node_mask_type=MaskType.attributes,
            edge_mask_type=MaskType.object,
        )

        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(data.a.size(0), dtype=torch.long, device=data.a.device)

        explanation = explainer(
            x=data.a,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            pos=data.pos,
            module_id=data.module_id,
            batch=batch,
            target=target,
        )

        edge_mask = explanation.edge_mask.detach().cpu().numpy().astype(np.float32) if explanation.edge_mask is not None else None
        node_mask = explanation.node_mask.detach().cpu().numpy().astype(np.float32) if explanation.node_mask is not None else None

        # Module aggregation
        module_id = data.module_id.detach().cpu().numpy().astype(np.int64)
        M = int(module_id.max() + 1)
        module_scores = np.zeros((M,), dtype=np.float32)
        if node_mask is not None:
            # node_mask may be [N,F] or [N]; reduce to [N]
            nm = node_mask
            if nm.ndim == 2:
                nm = nm.mean(axis=1)
            for m in range(M):
                module_scores[m] = float(nm[module_id == m].mean()) if np.any(module_id == m) else 0.0

        return {"edge_mask": edge_mask, "node_mask": node_mask, "module_scores": module_scores}

    def save(self, out_dir: str, subject_id: str, explanation: Dict, edge_index: torch.Tensor) -> None:
        out_dir = Path(out_dir) / str(subject_id)
        mkdirp(str(out_dir))
        if explanation.get("edge_mask") is not None:
            np.save(out_dir / "edge_mask.npy", explanation["edge_mask"])
            np.save(out_dir / "edge_index.npy", edge_index.detach().cpu().numpy().astype(np.int64))
        if explanation.get("node_mask") is not None:
            np.save(out_dir / "node_mask.npy", explanation["node_mask"])
        if explanation.get("module_scores") is not None:
            np.save(out_dir / "module_scores.npy", explanation["module_scores"])
        save_json({"subject_id": str(subject_id)}, str(out_dir / "explain_report.json"))
