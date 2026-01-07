from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max


def segment_softmax(scores: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Softmax over segments defined by index (e.g., batch vector).
    scores: [N]
    index:  [N] in [0..B-1]
    """
    max_per, _ = scatter_max(scores, index, dim=0)
    max_gather = max_per[index]
    exp = torch.exp(scores - max_gather)
    denom = scatter_add(exp, index, dim=0)
    return exp / (denom[index] + 1e-12)


class FusionAndClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
        )
        self.pool_W = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.pool_w = nn.Parameter(torch.zeros(2 * hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, num_classes),
        )

    def forward(self, H_intra: torch.Tensor, H_inter: torch.Tensor, batch: torch.Tensor) -> dict:
        Z = torch.cat([H_intra, H_inter], dim=-1)  # [N,2D]
        g = torch.sigmoid(self.gate(Z))           # [N,2D]
        H_fused = g * Z                           # [N,2D]

        s = torch.tanh(self.pool_W(H_fused)) @ self.pool_w  # [N]
        attn = segment_softmax(s, batch)                   # [N]
        graph_emb = scatter_add(H_fused * attn.unsqueeze(-1), batch, dim=0)  # [B,2D]
        logits = self.classifier(graph_emb)
        return {"logits": logits, "graph_emb": graph_emb, "pool_attn": attn, "H_fused": H_fused}
