import torch
import torch.nn as nn


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
        seq_length = q.shape[1]

        cos = self.cos_cached[:seq_length]
        sin = self.sin_cached[:seq_length]

        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

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
