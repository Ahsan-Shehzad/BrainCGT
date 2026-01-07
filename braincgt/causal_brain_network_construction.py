from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import Data

from .config import AppConfig
from .fmri_preprocessing import FMRI_Preprocessor
from .modular_parcellation import ModularParcellator
from .causal_connectivity_estimation import GrangerCausalityEstimator
from .utils import mkdirp, pca_first_component_score, save_json, to_edge_index


@dataclass
class BrainNetworkBuilder:
    cfg: AppConfig

    def __post_init__(self):
        self.preproc = FMRI_Preprocessor(self.cfg)
        self.parc = ModularParcellator(self.cfg)
        self.gc = GrangerCausalityEstimator(self.cfg)

    def build_subject(self, subj: Dict) -> Data:
        subject_id = str(subj["subject_id"])
        label = int(subj["label"])
        site_id = str(subj.get("site_id", "NA"))

        # Stage 1: preprocessing
        subj = self.preproc.run_subject(subj)
        preproc_path = subj["preproc_path"]

        # Stage 2: parcellation
        parc_out = self.parc.run(preproc_path, subject_id)
        X = parc_out["X"]              # [N,T]
        module_id = parc_out["module_id"]  # [N]
        pos = parc_out["pos"]          # [N,3]

        # Activity scalar per ROI (used in direction-aware embedding)
        a = pca_first_component_score(X)  # [N,1]

        # Stage 3: Granger causality -> adjacency
        gc_out = self.gc.fit_transform(X, module_id)
        A = gc_out["A"]  # [N,N]

        edge_index, edge_attr = to_edge_index(A)

        data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long),
        )
        data.a = torch.from_numpy(a.astype(np.float32))                  # [N,1]
        data.pos = torch.from_numpy(pos.astype(np.float32))              # [N,3]
        data.module_id = torch.from_numpy(module_id.astype(np.int64))    # [N]
        data.num_nodes = int(a.shape[0])

        # identifiers for traceability
        data.subject_id = subject_id
        data.site_id = site_id

        return data

    def build_all(self, subjects_df: pd.DataFrame, graphs_dir: str) -> pd.DataFrame:
        graphs_dir = Path(graphs_dir)
        mkdirp(str(graphs_dir))

        rows = []
        for _, row in subjects_df.iterrows():
            subj = row.to_dict()
            subject_id = str(subj["subject_id"])
            out_pt = graphs_dir / f"{subject_id}.pt"
            meta_json = graphs_dir / f"{subject_id}.json"
            if out_pt.exists():
                rows.append({
                    "subject_id": subject_id,
                    "label": int(subj["label"]),
                    "site_id": str(subj.get("site_id", "NA")),
                    "graph_path": str(out_pt),
                })
                continue

            try:
                data = self.build_subject(subj)
                torch.save(data, out_pt)
                save_json({
                    "subject_id": subject_id,
                    "site_id": str(subj.get("site_id", "NA")),
                    "label": int(subj["label"]),
                    "num_nodes": int(data.num_nodes),
                    "num_edges": int(data.edge_index.shape[1]),
                }, str(meta_json))
                rows.append({
                    "subject_id": subject_id,
                    "label": int(subj["label"]),
                    "site_id": str(subj.get("site_id", "NA")),
                    "graph_path": str(out_pt),
                })
                logger.info(f"Built graph for {subject_id}: nodes={data.num_nodes} edges={data.edge_index.shape[1]}\n")
            except Exception as e:
                logger.error(f"Failed to build graph for {subject_id}: {e}\n")

        index_df = pd.DataFrame(rows)
        index_path = graphs_dir / "graphs_index.csv"
        index_df.to_csv(index_path, index=False)
        return index_df


class GraphFolderDataset(torch.utils.data.Dataset):
    """
    Loads PyG Data objects from a folder and returns them directly.
    Expected: graphs_index.csv containing graph_path, label, site_id, subject_id.
    """
    def __init__(self, graphs_index_csv: str):
        self.df = pd.read_csv(graphs_index_csv)
        required = {"subject_id", "label", "site_id", "graph_path"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"graphs_index_csv must contain columns: {sorted(required)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Data:
        path = self.df.iloc[idx]["graph_path"]
        data: Data = torch.load(path)
        # ensure label is present
        if not hasattr(data, "y") or data.y is None:
            data.y = torch.tensor([int(self.df.iloc[idx]["label"])], dtype=torch.long)
        data.subject_id = str(self.df.iloc[idx]["subject_id"])
        data.site_id = str(self.df.iloc[idx]["site_id"])
        return data
