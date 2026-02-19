import torch


def build_conditioning_vector(
    z_past: torch.Tensor,
    a_past: torch.Tensor,
    q_last: torch.Tensor | None,
    use_proprio: bool,
) -> torch.Tensor:
    b = z_past.shape[0]
    z_flat = z_past.reshape(b, -1)
    a_flat = a_past.reshape(b, -1)

    if use_proprio:
        if q_last is None:
            raise ValueError("use_proprio=True but q_last is None")
        return torch.cat([z_flat, a_flat, q_last], dim=1)

    return torch.cat([z_flat, a_flat], dim=1)
