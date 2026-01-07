from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import softmax


class CausalSelfAttentionLayer(nn.Module):
    """
    Intra-module causal self-attention over directed edges (i -> j).
    For each query node i, neighbors are its outgoing edges i->j with module_id[i]==module_id[j].
    """
    def __init__(self, hidden_dim: int, num_heads: int, attn_dim: int, dropout: float, gamma: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.gamma = gamma

        self.q_proj = nn.Linear(hidden_dim, num_heads * attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * attn_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * attn_dim, hidden_dim, bias=False)

        # head-wise causal bias parameters
        self.w_a = nn.Parameter(torch.zeros(num_heads))
        self.b_a = nn.Parameter(torch.zeros(num_heads))

        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, module_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
          H: [N,D]
          edge_index: [2,E]
          edge_attr: [E,1]
          module_id: [N]
        """
        src = edge_index[0]  # query nodes i
        dst = edge_index[1]  # key/value nodes j
        intra_mask = (module_id[src] == module_id[dst])
        if intra_mask.sum() == 0:
            return H

        src = src[intra_mask]
        dst = dst[intra_mask]
        w = edge_attr[intra_mask].view(-1)  # [E_intra]

        q = self.q_proj(H).view(-1, self.num_heads, self.attn_dim)  # [N,H,d]
        k = self.k_proj(H).view(-1, self.num_heads, self.attn_dim)
        v = self.v_proj(H).view(-1, self.num_heads, self.attn_dim)

        q_e = q[src]  # [E,H,d]
        k_e = k[dst]
        v_e = v[dst]

        dot = (q_e * k_e).sum(dim=-1) / math.sqrt(self.attn_dim)  # [E,H]
        bias = torch.sigmoid(self.gamma * (w.unsqueeze(-1) * self.w_a.unsqueeze(0) + self.b_a.unsqueeze(0)))  # [E,H]
        logits = dot + bias

        # head-wise softmax over outgoing edges of each src
        alphas = []
        for h in range(self.num_heads):
            alphas.append(softmax(logits[:, h], index=src))
        alpha = torch.stack(alphas, dim=-1)  # [E,H]
        alpha = self.dropout(alpha)

        # weighted sum of v_e to each src node
        out = []
        for h in range(self.num_heads):
            msg = v_e[:, h, :] * alpha[:, h].unsqueeze(-1)  # [E,d]
            agg = scatter_add(msg, src, dim=0, dim_size=H.size(0))  # [N,d]
            out.append(agg)
        out = torch.cat(out, dim=-1)  # [N, H*d]
        out = self.out_proj(out)       # [N,D]

        H = self.ln1(H + out)
        H = self.ln2(H + self.ffn(H))
        return H
