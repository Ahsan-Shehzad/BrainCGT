from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .positional_encoding import fourier_positional_encoding
from .direction_aware_input_embeddings import DirectionAwareEmbedding
from .causal_self_attention_encoder import CausalSelfAttentionLayer
from .causal_cross_modular_encoders import CausalCrossAttentionLayer
from .hierarchical_adaptive_fusion_and_classification import FusionAndClassifier


class BrainCGT(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        attn_dim: int,
        dropout: float,
        gamma: float,
        use_posenc: bool,
        posenc_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.use_posenc = use_posenc
        self.posenc_dim = posenc_dim if use_posenc else 0

        self.embed = DirectionAwareEmbedding(hidden_dim, self.posenc_dim)

        self.intra_layers = nn.ModuleList([
            CausalSelfAttentionLayer(hidden_dim, num_heads, attn_dim, dropout, gamma)
            for _ in range(num_layers)
        ])
        self.inter_layers = nn.ModuleList([
            CausalCrossAttentionLayer(hidden_dim, num_heads, attn_dim, dropout, gamma)
            for _ in range(num_layers)
        ])

        # combine intra/inter for next layer
        self.update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.head = FusionAndClassifier(hidden_dim, num_classes, dropout)

    def forward(self, data: Data) -> dict:
        # Required fields: a [N,1], edge_index [2,E], edge_attr [E,1], pos [N,3], module_id [N], batch [N]
        a = data.a
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        pos = data.pos
        module_id = data.module_id
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(a.size(0), dtype=torch.long, device=a.device)

        pos_enc = fourier_positional_encoding(pos, self.posenc_dim) if self.use_posenc else pos.new_zeros((pos.size(0), 0))
        H = self.embed(a, edge_index, edge_attr, pos_enc)
        H0 = H

        H_intra = H
        H_inter = H
        for l in range(len(self.intra_layers)):
            H_intra = self.intra_layers[l](H, edge_index, edge_attr, module_id)
            H_inter = self.inter_layers[l](H_intra, edge_index, edge_attr, module_id)
            H = self.update[l](torch.cat([H_intra, H_inter], dim=-1))

        out = self.head(H_intra, H_inter, batch)
        logits = out["logits"]
        prob = torch.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "prob": prob,
            "H0": H0,
            "H_intra": H_intra,
            "H_inter": H_inter,
            "H_fused": out["H_fused"],
            "graph_emb": out["graph_emb"],
            "pool_attn": out["pool_attn"],
        }
