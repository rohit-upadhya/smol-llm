import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatoryPositionalEncoding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_length: int = 512,
        base: int = 10000,
    ):
        super().__init__()
        self.d_model = head_dim

        dim_pairs = torch.arange(0, head_dim, 2).float()

        inv_freq = 1.0 / (base ** (dim_pairs / head_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        m = torch.arange(max_seq_length, dtype=torch.float32)

        angles = torch.outer(m, self.inv_freq)

        angles = torch.cat((angles, angles), dim=-1)

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self,
        q,
        k,
    ):
        seq_length = q.shape[2]

        cos = self.cos_cached[:seq_length]
        sin = self.sin_cached[:seq_length]

        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)

        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated

    def _rotate_half(
        self,
        x,
    ):
        x_1, x_2 = torch.chunk(x, 2, dim=-1)

        x = torch.cat((-x_2, x_1), dim=-1)
        return x


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.epselon = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        pass

    def forward(
        self,
        x,
    ):
        x_squared = torch.pow(x, 2)
        squared_mean = torch.mean(x_squared, dim=-1)
        root_squared_mean = torch.sqrt(squared_mean + self.epselon)

        y = (x / root_squared_mean.unsqueeze(-1)) * self.gamma

        return y


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.hidden_dim = int((8 / 3) * dim)
        self.ff_up = nn.Sequential(
            nn.Linear(dim, self.hidden_dim, bias=False),
        )
        self.ff_gate = nn.Sequential(
            nn.Linear(dim, self.hidden_dim, bias=False),
            nn.SiLU(),
        )

        self.ff_down = nn.Sequential(
            nn.Linear(self.hidden_dim, dim, bias=False),
        )

    def forward(
        self,
        x,
    ):
        x_up = self.ff_up(x)
        x_gate = self.ff_gate(x)

        x = self.ff_down(x_up * x_gate)

        return x


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        self.head_dim = dim // n_heads

        self.k_proj = nn.Linear(dim, dim, bias=False)

        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.q_proj = nn.Linear(dim, dim, bias=False)

        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.rotatory_emb = RotatoryPositionalEncoding(head_dim=self.head_dim)

    def forward(
        self,
        x,
    ):
        B, SEQ, DIM = x.shape
        k = self.k_proj(x).reshape(B, SEQ, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_proj(x).reshape(B, SEQ, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, SEQ, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotatory_emb(q, k)

        attention_score = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        mask = torch.triu(torch.ones_like(attention_score), diagonal=1).bool()

        attention_score = attention_score.masked_fill(mask, -float("inf"))

        attention_weights = F.softmax(attention_score, dim=-1)

        v = attention_weights @ v

        attention_output = self.o_proj(
            v.transpose(1, 2).contiguous().reshape(B, SEQ, DIM)
        )

        return attention_output
