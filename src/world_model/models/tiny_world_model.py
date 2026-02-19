import torch
import torch.nn as nn

from world_model.conditioning import build_conditioning_vector


class TinyLatentWorldModel(nn.Module):
    """
    Minimal predictor for end-to-end pipeline validation.
    Predicts future latents from past latents/actions and optional proprio.
    """

    def __init__(
        self,
        z_dim: int,
        a_dim: int,
        q_dim: int,
        context_len: int,
        horizon_len: int,
        hidden: int,
        use_proprio: bool,
    ):
        super().__init__()
        self.use_proprio = use_proprio
        self.horizon_len = horizon_len
        self.z_dim = z_dim

        cond_dim = (context_len * z_dim) + (context_len * a_dim)
        if use_proprio:
            cond_dim += q_dim

        out_dim = horizon_len * z_dim

        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z_past: torch.Tensor, a_past: torch.Tensor, q_last: torch.Tensor | None) -> torch.Tensor:
        x = build_conditioning_vector(
            z_past=z_past,
            a_past=a_past,
            q_last=q_last,
            use_proprio=self.use_proprio,
        )
        y = self.net(x)
        return y.reshape(z_past.shape[0], self.horizon_len, self.z_dim)
