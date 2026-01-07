from __future__ import annotations

import numpy as np
import torch


def fourier_positional_encoding(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fourier feature positional encoding from MNI coordinates.

    Args:
      pos: [N,3] coordinates in mm
      dim: output dimension, must be divisible by 6

    Returns:
      pos_enc: [N,dim]
    """
    if dim == 0:
        return torch.zeros((pos.shape[0], 0), device=pos.device, dtype=pos.dtype)
    if dim % 6 != 0:
        raise ValueError("posenc_dim must be divisible by 6 for (sin,cos) x (x,y,z).")

    # Normalize coordinates to approximately [-1,1] using typical MNI bounds
    # (This is stable across cohorts and avoids dataset leakage.)
    bounds = torch.tensor([90.0, 126.0, 72.0], device=pos.device, dtype=pos.dtype)  # |x|<=90, |y|<=126, |z|<=72
    p = torch.clamp(pos / bounds, min=-1.0, max=1.0)

    n_freq = dim // 6
    # Log-spaced frequencies
    freq = torch.logspace(0.0, 3.0, steps=n_freq, device=pos.device, dtype=pos.dtype)  # [n_freq]
    # [N,3,n_freq]
    p_exp = p.unsqueeze(-1) * freq
    ang = 2.0 * torch.pi * p_exp

    sin = torch.sin(ang)
    cos = torch.cos(ang)

    # concatenate in order: x_sin,x_cos,y_sin,y_cos,z_sin,z_cos
    x_sin, y_sin, z_sin = sin[:, 0, :], sin[:, 1, :], sin[:, 2, :]
    x_cos, y_cos, z_cos = cos[:, 0, :], cos[:, 1, :], cos[:, 2, :]

    out = torch.cat([x_sin, x_cos, y_sin, y_cos, z_sin, z_cos], dim=-1)
    return out
