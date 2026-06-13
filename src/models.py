"""Noise-prediction network for the 2D DDPM teacher."""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding of the timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2  # 32 in our case
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MLPDenoiser(nn.Module):
    """Small MLP that predicts the noise eps given (x_t, t)."""

    def __init__(self, data_dim=2, hidden_dim=128, n_layers=4, time_embed_dim=64):
        super().__init__()
        self.config = {
            "data_dim": data_dim,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "time_embed_dim": time_embed_dim,
        }

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = []
        in_dim = data_dim + hidden_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        temb = self.time_mlp(self.time_embed(t))
        h = torch.cat([x, temb], dim=-1)
        return self.out(self.net(h))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
