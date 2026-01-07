from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_add


class DirectionAwareEmbedding(nn.Module):
    """
    Direction-aware node embedding:
      f_i = [a_i, d_in(i), d_out(i), pos_enc_i] -> Linear -> h_i^{(0)} in R^D
    """
    def __init__(self, hidden_dim: int, pos_dim: int):
        super().__init__()
        in_dim = 1 + 1 + 1 + pos_dim
        self.proj = nn.Linear(in_dim, hidden_dim)

    @staticmethod
    def _degrees(edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int):
        # edge_index: [2,E], edge_attr: [E,1]
        w = edge_attr.view(-1)
        src = edge_index[0]
        dst = edge_index[1]
        d_out = scatter_add(w, src, dim=0, dim_size=num_nodes)
        d_in = scatter_add(w, dst, dim=0, dim_size=num_nodes)
        return d_in.unsqueeze(-1), d_out.unsqueeze(-1)

    def forward(self, a: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
          a: [N,1] activity scalar per ROI
          edge_index: [2,E]
          edge_attr: [E,1] causal weights in [0,1]
          pos_enc: [N,pos_dim]
        """
        N = a.shape[0]
        d_in, d_out = self._degrees(edge_index, edge_attr, N)
        f = torch.cat([a, d_in, d_out, pos_enc], dim=-1)
        return self.proj(f)
