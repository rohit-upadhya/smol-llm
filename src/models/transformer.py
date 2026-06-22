import torch
import torch.nn as nn

from src.models.layers import (
    RMSNorm,
    CausalSelfAttention,
    SwiGLU,
)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
    ):

        super().__init__()
        self.pre_rmsnorm = RMSNorm(dim=dim)
        self.post_rmsnorm = RMSNorm(dim=dim)
        self.csa = CausalSelfAttention(dim=dim, n_heads=n_heads)
        self.swiglu = SwiGLU(dim=dim)

    def forward(
        self,
        x,
    ):
        x_new = self.pre_rmsnorm(x)
        x_new = self.csa(x_new)
        x = x + x_new

        x_new = self.post_rmsnorm(x)
        x_new = self.swiglu(x_new)
        x = x + x_new

        return x
