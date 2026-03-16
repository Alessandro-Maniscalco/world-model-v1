"""Tests for chunkwise flow matching over structured Wan VACE latent videos."""

from __future__ import annotations

import torch
import torch.nn as nn

from world_model.models.wan_vace_conditioning import ActionTokenEncoder
from world_model.training import train_chunkwise_batch
from world_model.training.flow_matching import chunkwise_teacher_forcing_loss


class _RecordingVideoModel(nn.Module):
    """Record Wan VACE-style inputs and echo the noisy future video."""

    def __init__(self) -> None:
        """Initialize learnable scale and call storage."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        *,
        noisy_future_video: torch.Tensor,
        observed_video: torch.Tensor,
        action_tokens: torch.Tensor,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Record structured video-path inputs and return a scaled prediction."""
        del action_tokens, observed_mask, control_hidden_states_scale
        self.calls.append(
            {
                "observed_frames": torch.tensor(observed_video.shape[2]),
                "future_frames": torch.tensor(noisy_future_video.shape[2]),
                "mask": block_causal_attention_mask.detach().clone(),
                "timestep_t": timestep_t.detach().clone(),
            }
        )
        return self.scale * noisy_future_video


def test_chunkwise_teacher_forcing_loss_supports_structured_latent_videos() -> None:
    """Compute flow-matching loss over structured latent videos and backpropagate."""
    torch.manual_seed(0)
    model = _RecordingVideoModel()
    z_past_video = torch.randn(2, 16, 3, 8, 8)
    z_future_video = torch.randn(2, 16, 8, 8, 8)
    action_tokens = torch.randn(2, 8, 4096)

    info = chunkwise_teacher_forcing_loss(
        model,
        z_past_video=z_past_video,
        z_future_video=z_future_video,
        action_tokens=action_tokens,
        k=1,
        t_min=0.3,
        t_max=0.3,
        return_info=True,
    )
    info.loss.backward()

    assert len(model.calls) == 2
    assert info.per_chunk_lengths == (4, 4)
    assert model.calls[0]["observed_frames"].item() == 3
    assert model.calls[1]["observed_frames"].item() == 7
    assert model.calls[0]["future_frames"].item() == 8
    assert model.calls[1]["future_frames"].item() == 4
    assert model.calls[0]["mask"].shape == (11, 11)
    assert model.calls[1]["mask"].shape == (11, 11)
    assert torch.allclose(model.calls[0]["timestep_t"], torch.full((2,), 300.0))
    assert torch.allclose(model.calls[1]["timestep_t"], torch.full((2,), 300.0))
    assert model.scale.grad is not None
    assert model.scale.grad.abs().item() > 0.0


def test_train_chunkwise_batch_supports_structured_latent_videos() -> None:
    """Run one optimizer step over the structured Wan VACE latent-video path."""
    torch.manual_seed(0)
    model = _RecordingVideoModel()
    action_encoder = ActionTokenEncoder(action_dim=6, hidden_dim=16)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(action_encoder.parameters()),
        lr=1e-2,
    )
    z_past_video = torch.randn(2, 16, 3, 8, 8)
    z_future_video = torch.randn(2, 16, 8, 8, 8)
    a_plan = torch.randn(2, 8, 6)

    metrics = train_chunkwise_batch(
        model=model,
        action_encoder=action_encoder,
        optimizer=optimizer,
        z_past_video=z_past_video,
        z_future_video=z_future_video,
        a_plan=a_plan,
        k=1,
        t_min=0.5,
        t_max=0.5,
    )

    assert metrics.loss > 0.0
    assert metrics.grad_norm > 0.0
    assert metrics.per_chunk_lengths == (4, 4)
    assert len(model.calls) == 2


class _ActionTokenOnlyVideoModel(nn.Module):
    """Drive the structured video loss entirely from action tokens."""

    def forward(
        self,
        *,
        noisy_future_video: torch.Tensor,
        observed_video: torch.Tensor,
        action_tokens: torch.Tensor,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project active-window action tokens into the leading latent frames."""
        del observed_video, timestep_t, block_causal_attention_mask, observed_mask, control_hidden_states_scale
        channels = noisy_future_video.shape[1]
        token_steps = action_tokens.shape[1]
        prediction = torch.zeros_like(noisy_future_video)
        prediction[:, :, :token_steps, :, :] = (
            action_tokens[:, :, :channels]
            .permute(0, 2, 1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, noisy_future_video.shape[3], noisy_future_video.shape[4])
        )
        return prediction


def test_train_chunkwise_batch_structured_video_grad_norm_includes_action_encoder() -> None:
    """Include action-token encoder gradients in structured-video grad accounting."""
    torch.manual_seed(0)
    model = _ActionTokenOnlyVideoModel()
    action_encoder = ActionTokenEncoder(action_dim=6, hidden_dim=16)
    optimizer = torch.optim.AdamW(action_encoder.parameters(), lr=1e-2)

    metrics = train_chunkwise_batch(
        model=model,
        action_encoder=action_encoder,
        optimizer=optimizer,
        z_past_video=torch.randn(2, 16, 3, 8, 8),
        z_future_video=torch.randn(2, 16, 8, 8, 8),
        a_plan=torch.randn(2, 8, 6),
        k=1,
        t_min=0.5,
        t_max=0.5,
    )

    assert metrics.loss > 0.0
    assert metrics.grad_norm > 0.0


def test_train_chunkwise_batch_reports_unclipped_grad_norm_when_disabled() -> None:
    """Use the explicit unclipped grad-norm branch when gradient clipping is disabled."""
    torch.manual_seed(0)
    model = _RecordingVideoModel()
    action_encoder = ActionTokenEncoder(action_dim=6, hidden_dim=16)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(action_encoder.parameters()),
        lr=1e-2,
    )

    metrics = train_chunkwise_batch(
        model=model,
        action_encoder=action_encoder,
        optimizer=optimizer,
        z_past_video=torch.randn(2, 16, 3, 8, 8),
        z_future_video=torch.randn(2, 16, 8, 8, 8),
        a_plan=torch.randn(2, 8, 6),
        k=1,
        t_min=0.5,
        t_max=0.5,
        grad_clip_norm=None,
    )

    assert metrics.loss > 0.0
    assert metrics.grad_norm > 0.0
