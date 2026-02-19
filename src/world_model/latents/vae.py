import torch


def normalize_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if float(x.max().cpu()) > 1.5:
            x = x / 255.0
    return x * 2.0 - 1.0


def to_bcthw(video: torch.Tensor) -> torch.Tensor:
    # Accept T,C,H,W or T,H,W,C and return B,C,T,H,W
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got {video.ndim}D {tuple(video.shape)}")
    if video.shape[1] == 3:
        tchw = video
    elif video.shape[-1] == 3:
        tchw = video.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unrecognized video shape: {tuple(video.shape)}")
    return tchw.permute(1, 0, 2, 3).unsqueeze(0)


@torch.no_grad()
def encode_window_to_latents(vae, video_window_tchw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    video_window_tchw: [B, T, C, H, W] or [B, T, H, W, C]
    returns latents: [B, C_lat, T_lat, H_lat, W_lat]
    """
    if video_window_tchw.ndim != 5:
        raise ValueError(f"Expected 5D batched video, got {video_window_tchw.ndim}D {tuple(video_window_tchw.shape)}")

    if video_window_tchw.shape[2] == 3:
        bcthw = video_window_tchw.permute(0, 2, 1, 3, 4)
    elif video_window_tchw.shape[-1] == 3:
        btchw = video_window_tchw.permute(0, 1, 4, 2, 3)
        bcthw = btchw.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Unrecognized batched video shape: {tuple(video_window_tchw.shape)}")

    bcthw = normalize_to_minus1_1(bcthw).to(device)
    enc = vae.encode(bcthw)
    return enc.latent_dist.mean
