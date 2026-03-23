"""Tests for the local-repo resolution sweep smoke-check script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_script_module():
    """Load the local-repo resolution sweep script from its file path."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "check" / "sweep_local_repo_resolutions.py"
    spec = importlib.util.spec_from_file_location("sweep_local_repo_resolutions", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script_module()


def test_checkpoint_single_resolution_uses_run_stem_and_sweep_local_root() -> None:
    """Checkpoint sweeps should default to a run-named MP4 under runs/sweep_local."""
    output_path, comparison_path, summary_path = script._resolve_output_artifacts(
        mode="checkpoint",
        output_dir=None,
        checkpoint_path=Path(
            "runs/test_full_multi_320x240_lora8_none/checkpoints/step_0000800.pt"
        ),
        label="320x240",
        resolution_count=1,
    )

    assert output_path == Path(
        "runs/sweep_local/test_full_multi_320x240_lora8_none_step_0000800.mp4"
    )
    assert comparison_path == Path(
        "runs/sweep_local/test_full_multi_320x240_lora8_none_step_0000800_comparison.mp4"
    )
    assert summary_path == Path(
        "runs/sweep_local/test_full_multi_320x240_lora8_none_step_0000800_summary.json"
    )


def test_checkpoint_multi_resolution_adds_label_suffix() -> None:
    """Checkpoint sweeps should suffix the resolution when multiple sizes are requested."""
    output_path, comparison_path, summary_path = script._resolve_output_artifacts(
        mode="checkpoint",
        output_dir=Path("custom_out"),
        checkpoint_path=Path("runs/example_run/checkpoints/step_0000100.pt"),
        label="384x288",
        resolution_count=2,
    )

    assert output_path == Path("custom_out/example_run_step_0000100_384x288.mp4")
    assert comparison_path == Path("custom_out/example_run_step_0000100_384x288_comparison.mp4")
    assert summary_path == Path("custom_out/example_run_step_0000100_summary.json")


def test_base_mode_keeps_resolution_named_outputs() -> None:
    """Base sweeps should keep resolution-named outputs under the shared sweep root."""
    output_path, comparison_path, summary_path = script._resolve_output_artifacts(
        mode="base",
        output_dir=None,
        checkpoint_path=None,
        label="512x384",
        resolution_count=1,
    )

    assert output_path == Path("runs/sweep_local/512x384.mp4")
    assert comparison_path == Path(
        "runs/sweep_local/512x384_comparison.mp4"
    )
    assert summary_path == Path("runs/sweep_local/summary.json")


def test_repo_prompt_mode_keeps_resolution_named_outputs() -> None:
    """Checkpoint-free repo prompt sweeps should use shared resolution-named outputs."""
    output_path, comparison_path, summary_path = script._resolve_output_artifacts(
        mode="repo_prompt",
        output_dir=None,
        checkpoint_path=None,
        label="512x384",
        resolution_count=1,
    )

    assert output_path == Path("runs/sweep_local/512x384.mp4")
    assert comparison_path == Path("runs/sweep_local/512x384_comparison.mp4")
    assert summary_path == Path("runs/sweep_local/summary.json")


def test_base_mode_forwards_prompt_and_guidance_to_local_pipeline(monkeypatch, tmp_path) -> None:
    """Base-mode sweeps should pass the caller prompt and guidance into local pipeline inference."""
    captured_local_kwargs: dict[str, object] = {}
    target_rollout = torch.full((1, 9, 3, 4, 5), 11.0)

    monkeypatch.setattr(
        script,
        "_load_base_runtime_config",
        lambda path: SimpleNamespace(
            control_scale=1.0,
            max_sequence_length=128,
            prompt="default prompt",
            guidance_scale=1.0,
        ),
    )
    monkeypatch.setattr(script, "_resolve_device", lambda device_name: torch.device("cpu"))
    monkeypatch.setattr(script, "_select_runtime_dtype", lambda device: torch.float32)
    monkeypatch.setattr(
        script,
        "_load_checkpoint_clip",
        lambda **kwargs: (torch.zeros(1, 9, 3, 240, 320), torch.zeros(9, 1)),
    )
    monkeypatch.setattr(
        script,
        "preprocess_video_for_vae",
        lambda video, frame_height, frame_width: target_rollout,
    )
    monkeypatch.setattr(
        script,
        "_build_dense_prefix_condition_lists",
        lambda **kwargs: (["video"], ["mask"]),
    )
    monkeypatch.setattr(script, "_load_local_pipeline", lambda **kwargs: object())

    def _fake_run_local_pipeline(**kwargs):
        """Capture the effective prompt-conditioning args passed into base-mode inference."""
        captured_local_kwargs.update(kwargs)
        return torch.full((9, 4, 5, 3), 22.0, dtype=torch.float32).numpy()

    monkeypatch.setattr(script, "_run_local_pipeline", _fake_run_local_pipeline)
    monkeypatch.setattr(script, "_tensor_video_to_frames", lambda video_btchw: [0.0])
    monkeypatch.setattr(script, "_build_side_by_side_video", lambda **kwargs: torch.zeros(1, 9, 3, 4, 5))
    monkeypatch.setattr(script, "_export_video", lambda **kwargs: None)
    monkeypatch.setattr(script, "_write_plausibility_report", lambda **kwargs: {"plausible": True})
    monkeypatch.setattr(
        script,
        "_write_arm_motion_report",
        lambda **kwargs: {"summary": {"motion_verdict": "good"}},
    )

    result = script._run_one_checkpoint_resolution(
        mode="base",
        config_path=Path("configs/train/world_model.yaml"),
        checkpoint_path=Path("unused.pt"),
        width=320,
        height=240,
        output_path=tmp_path / "generated.mp4",
        comparison_path=tmp_path / "comparison.mp4",
        plausibility_output_path=tmp_path / "plausibility_report.json",
        motion_output_path=tmp_path / "arm_motion_report.json",
        repo_id="repo",
        episode_index=1,
        start_frame=60,
        video_key="observation.images.cam_high",
        context_len=9,
        horizon_len=8,
        k=1,
        integration_steps=50,
        fps=10,
        seed=0,
        single_chunk_rollout=True,
        device_name="cpu",
        action_source="auto",
        action_scale=1.0,
        action_token_scale=1.0,
        control_scale=None,
        prompt="robot arm picks up a fork from a table",
        negative_prompt="",
        guidance_scale=3.5,
        max_sequence_length=256,
    )

    assert captured_local_kwargs["prompt"] == "robot arm picks up a fork from a table"
    assert captured_local_kwargs["negative_prompt"] == ""
    assert captured_local_kwargs["guidance_scale"] == 3.5
    assert captured_local_kwargs["max_sequence_length"] == 256
    assert captured_local_kwargs["conditioning_scale"] == 1.0
    assert result["plausibility"] == {"plausible": True}
    assert result["motion"] == {"summary": {"motion_verdict": "good"}}


def test_repo_prompt_mode_forwards_prompt_to_repo_world_model(monkeypatch, tmp_path) -> None:
    """Repo prompt sweeps should load base runtime config and call repo inference without a checkpoint."""
    captured_runtime_cfg: list[SimpleNamespace] = []
    captured_checkpoints: list[object] = []

    monkeypatch.setattr(
        script,
        "_load_base_runtime_config",
        lambda path: SimpleNamespace(
            control_scale=1.0,
            max_sequence_length=128,
            prompt="default prompt",
            negative_prompt="",
            guidance_scale=1.0,
            context_len=9,
            horizon_len=8,
            conditioning_mode="none",
            action_token_scale=1.0,
            wan_vace_model_id="wan",
            chunk_schedule_mode="k_chunks",
            future_latent_residual_mode="none",
        ),
    )
    monkeypatch.setattr(script, "_resolve_device", lambda device_name: torch.device("cpu"))
    monkeypatch.setattr(script, "_select_runtime_dtype", lambda device: torch.float32)
    monkeypatch.setattr(
        script,
        "_load_checkpoint_clip",
        lambda **kwargs: (torch.zeros(1, 17, 3, 240, 320), torch.zeros(1, 17, 2)),
    )

    def _fake_run_checkpoint_world_model(*, runtime_cfg, checkpoint, **kwargs):
        """Capture prompt-conditioned runtime config and checkpoint usage."""
        captured_runtime_cfg.append(runtime_cfg)
        captured_checkpoints.append(checkpoint)
        target = torch.zeros(1, 17, 3, 4, 5)
        pred = torch.ones(1, 17, 3, 4, 5)
        return target, pred

    monkeypatch.setattr(script, "_run_checkpoint_world_model", _fake_run_checkpoint_world_model)
    monkeypatch.setattr(script, "_tensor_video_to_frames", lambda video_btchw: [0.0])
    monkeypatch.setattr(script, "_build_side_by_side_video", lambda **kwargs: torch.zeros(1, 17, 3, 4, 5))
    monkeypatch.setattr(script, "_export_video", lambda **kwargs: None)
    monkeypatch.setattr(script, "_write_plausibility_report", lambda **kwargs: {"plausible": True})
    monkeypatch.setattr(
        script,
        "_write_arm_motion_report",
        lambda **kwargs: {"summary": {"motion_verdict": "good"}},
    )

    result = script._run_one_checkpoint_resolution(
        mode="repo_prompt",
        config_path=Path("configs/train/world_model.yaml"),
        checkpoint_path=Path("unused.pt"),
        width=320,
        height=240,
        output_path=tmp_path / "generated.mp4",
        comparison_path=tmp_path / "comparison.mp4",
        plausibility_output_path=tmp_path / "plausibility_report.json",
        motion_output_path=tmp_path / "arm_motion_report.json",
        repo_id="repo",
        episode_index=1,
        start_frame=60,
        video_key="observation.images.cam_high",
        context_len=9,
        horizon_len=8,
        k=1,
        integration_steps=50,
        fps=10,
        seed=0,
        single_chunk_rollout=True,
        device_name="cpu",
        action_source="auto",
        action_scale=1.0,
        action_token_scale=1.0,
        control_scale=None,
        prompt="robot arm picks up a fork from a table",
        negative_prompt="distorted colors",
        guidance_scale=5.0,
        max_sequence_length=256,
    )

    assert len(captured_runtime_cfg) == 1
    runtime_cfg = captured_runtime_cfg[0]
    assert runtime_cfg.conditioning_mode == "prompt"
    assert runtime_cfg.prompt == "robot arm picks up a fork from a table"
    assert runtime_cfg.negative_prompt == "distorted colors"
    assert runtime_cfg.guidance_scale == 5.0
    assert runtime_cfg.max_sequence_length == 256
    assert runtime_cfg.single_chunk_rollout is True
    assert captured_checkpoints == [None]
    assert "Repo prompt mode uses the repo chunkwise world-model inference path with prompt tokens and no checkpoint overlay." in result["notes"]
    assert result["plausibility"] == {"plausible": True}
    assert result["motion"] == {"summary": {"motion_verdict": "good"}}


def test_run_local_pipeline_enables_cfg_when_guidance_scale_above_one() -> None:
    """Local base inference should request and apply CFG prompt embeddings when enabled."""

    class _FakeProgressBar:
        """Minimal progress-bar stub for local-pipeline tests."""

        def __init__(self) -> None:
            self.description = None
            self.updates = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def set_description(self, description: str) -> None:
            self.description = description

        def update(self, count: int = 1) -> None:
            self.updates += count

    class _FakeScheduler:
        """Minimal scheduler stub that exposes one denoising step."""

        def __init__(self) -> None:
            self.timesteps = torch.tensor([1.0], dtype=torch.float32)

        def set_timesteps(self, num_inference_steps: int, device: torch.device) -> None:
            del num_inference_steps, device

        def step(self, noise_pred: torch.Tensor, timestep: torch.Tensor, latents: torch.Tensor, return_dict: bool):
            del timestep, return_dict
            return (latents - noise_pred,)

    class _FakeTransformer:
        """Transformer stub that distinguishes conditioned and unconditioned branches."""

        def __init__(self) -> None:
            self.dtype = torch.float32
            self.config = SimpleNamespace(vace_layers=[0], in_channels=2)
            self.encoder_hidden_states_calls: list[torch.Tensor] = []

        def __call__(
            self,
            *,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            control_hidden_states: torch.Tensor,
            control_hidden_states_scale: torch.Tensor,
            attention_kwargs,
            return_dict: bool,
        ):
            del timestep, control_hidden_states, control_hidden_states_scale, attention_kwargs, return_dict
            self.encoder_hidden_states_calls.append(encoder_hidden_states.clone())
            fill = float(encoder_hidden_states.mean().item())
            return (torch.full_like(hidden_states, fill),)

    class _FakeVAE:
        """VAE stub that decodes latents into a tiny RGB video."""

        def __init__(self) -> None:
            self.dtype = torch.float32
            self.config = SimpleNamespace(latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0], z_dim=2)

        def decode(self, latents: torch.Tensor, return_dict: bool):
            del return_dict
            batch, _, frames, height, width = latents.shape
            return (torch.zeros((batch, 3, frames, height, width), dtype=torch.float32),)

    class _FakeVideoProcessor:
        """Video postprocessor stub that returns BTCHW tensors as numpy arrays."""

        def postprocess_video(self, video: torch.Tensor, output_type: str):
            assert output_type == "np"
            return video.permute(0, 2, 3, 4, 1).cpu().numpy()

    class _FakePipe:
        """Pipeline stub exposing only the local-pipeline methods used by the smoke test."""

        def __init__(self) -> None:
            self.vae_scale_factor_temporal = 4
            self._execution_device = torch.device("cpu")
            self.transformer = _FakeTransformer()
            self.vae = _FakeVAE()
            self.scheduler = _FakeScheduler()
            self.video_processor = _FakeVideoProcessor()
            self.encode_prompt_call: dict[str, object] | None = None

        def encode_prompt(self, **kwargs):
            self.encode_prompt_call = kwargs
            prompt_embeds = torch.ones((1, 4, 3), dtype=torch.float32)
            negative_prompt_embeds = torch.zeros((1, 4, 3), dtype=torch.float32)
            return prompt_embeds, negative_prompt_embeds

        def preprocess_conditions(self, *args):
            del args
            return (
                torch.zeros((1, 3, 9, 2, 2), dtype=torch.float32),
                torch.zeros((1, 1, 9, 2, 2), dtype=torch.float32),
                [[]],
            )

        def prepare_video_latents(self, *args):
            del args
            return torch.zeros((1, 2, 9, 2, 2), dtype=torch.float32)

        def prepare_masks(self, *args):
            del args
            return torch.zeros((1, 1, 9, 2, 2), dtype=torch.float32)

        def prepare_latents(
            self,
            batch_size: int,
            in_channels: int,
            height: int,
            width: int,
            num_frames: int,
            dtype: torch.dtype,
            device: torch.device,
            generator,
            latents,
        ) -> torch.Tensor:
            del batch_size, height, width, generator, latents
            return torch.zeros((1, in_channels, num_frames, 2, 2), dtype=dtype, device=device)

        def progress_bar(self, total: int) -> _FakeProgressBar:
            del total
            return _FakeProgressBar()

        def maybe_free_model_hooks(self) -> None:
            return None

    fake_pipe = _FakePipe()
    output = script._run_local_pipeline(
        pipe=fake_pipe,
        video_frames=["video"],
        mask_frames=["mask"],
        height=32,
        width=32,
        num_frames=9,
        num_inference_steps=1,
        generator=None,
        guidance_scale=5.0,
        max_sequence_length=128,
        conditioning_scale=1.0,
        prompt="pick up the fork",
        negative_prompt="distorted colors",
        progress_label="debug",
    )

    assert fake_pipe.encode_prompt_call is not None
    assert fake_pipe.encode_prompt_call["do_classifier_free_guidance"] is True
    assert fake_pipe.encode_prompt_call["negative_prompt"] == "distorted colors"
    assert len(fake_pipe.transformer.encoder_hidden_states_calls) == 2
    assert torch.allclose(fake_pipe.transformer.encoder_hidden_states_calls[0], torch.ones((1, 4, 3)))
    assert torch.allclose(fake_pipe.transformer.encoder_hidden_states_calls[1], torch.zeros((1, 4, 3)))
    assert output.shape == (9, 2, 2, 3)


def test_single_resolution_plausibility_path_uses_shared_name() -> None:
    """Single-resolution sweeps should emit the controller-expected plausibility filename."""
    plausibility_path = script._resolve_plausibility_output_path(
        output_path=Path("runs/example/generated.mp4"),
        resolution_count=1,
    )

    assert plausibility_path == Path("runs/example/plausibility_report.json")


def test_multi_resolution_plausibility_path_uses_video_stem() -> None:
    """Multi-resolution sweeps should keep per-video plausibility filenames distinct."""
    plausibility_path = script._resolve_plausibility_output_path(
        output_path=Path("runs/example/generated_384x288.mp4"),
        resolution_count=2,
    )

    assert plausibility_path == Path("runs/example/generated_384x288_plausibility_report.json")


def test_single_resolution_motion_path_uses_shared_name() -> None:
    """Single-resolution sweeps should emit the shared arm-motion filename."""
    motion_path = script._resolve_motion_output_path(
        output_path=Path("runs/example/generated.mp4"),
        resolution_count=1,
    )

    assert motion_path == Path("runs/example/arm_motion_report.json")


def test_write_plausibility_report_writes_json(tmp_path) -> None:
    """Writing a plausibility report should succeed through the real checker module."""
    grid = torch.linspace(0.2, 0.8, steps=16, dtype=torch.float32).reshape(4, 4)
    frame0 = torch.stack([grid, grid.transpose(0, 1), torch.flip(grid, dims=(0,))], dim=0)
    frame1 = torch.roll(frame0, shifts=1, dims=2)
    video = torch.stack([frame0, frame1], dim=0).unsqueeze(0)
    output_json = tmp_path / "plausibility_report.json"

    script._PLAUSIBILITY_MODULE = None
    summary = script._write_plausibility_report(
        reference_video=video,
        generated_video=video.clone(),
        output_json=output_json,
    )

    report = json.loads(output_json.read_text())
    assert summary["plausible"] is True
    assert output_json.exists()
    assert report["summary"]["plausible"] is True
    assert len(report["frames"]) == 2


def test_decode_future_latents_uses_past_context_to_keep_full_wan_horizon() -> None:
    """Decode the full latent window so a one-step Wan future still reconstructs four RGB frames."""

    class _FakeWanVAE:
        def __init__(self) -> None:
            self.vae = SimpleNamespace(to=lambda device=None, dtype=None: self.vae)

        def decode(self, latents, output_layout="BTCHW", output_range="zero_to_one"):
            del output_layout, output_range
            frame_count = 1 + ((int(latents.shape[2]) - 1) * 4)
            frame_ids = torch.arange(frame_count, dtype=torch.float32).view(1, frame_count, 1, 1, 1)
            return frame_ids.expand(latents.shape[0], frame_count, 3, 1, 1)

    pred_video, target_video = script._decode_future_latents(
        vae=_FakeWanVAE(),
        past_video_latents=torch.zeros(1, 16, 6, 1, 1),
        pred_future_video=torch.zeros(1, 16, 1, 1, 1),
        target_future_video=torch.zeros(1, 16, 1, 1, 1),
        context_len=21,
        future_frame_count=4,
        device=torch.device("cpu"),
        runtime_dtype=torch.float32,
    )

    assert pred_video.shape == (1, 4, 3, 1, 1)
    assert target_video.shape == (1, 4, 3, 1, 1)
    assert torch.equal(pred_video[0, :, 0, 0, 0], torch.tensor([21.0, 22.0, 23.0, 24.0]))
    assert torch.equal(target_video[0, :, 0, 0, 0], torch.tensor([21.0, 22.0, 23.0, 24.0]))


def test_checkpoint_mode_exports_generated_rollout_and_comparison_order(monkeypatch, tmp_path) -> None:
    """Checkpoint sweeps should export the generated rollout and keep it on the comparison right."""
    target_rollout = torch.full((1, 2, 3, 4, 5), 11.0)
    pred_rollout = torch.full((1, 2, 3, 4, 5), 22.0)
    export_calls: list[dict[str, object]] = []
    plausibility_calls: list[dict[str, object]] = []
    motion_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        script,
        "_load_checkpoint_runtime_config",
        lambda path: (
            {},
            SimpleNamespace(
                frame_width=320,
                frame_height=240,
                trainable_backbone="full",
                conditioning_mode="none",
                context_len=9,
                horizon_len=8,
            ),
        ),
    )
    monkeypatch.setattr(script, "_resolve_device", lambda device_name: torch.device("cpu"))
    monkeypatch.setattr(script, "_select_runtime_dtype", lambda device: torch.float32)
    monkeypatch.setattr(
        script,
        "_load_checkpoint_clip",
        lambda **kwargs: (torch.zeros(1, 17, 3, 240, 320), torch.zeros(17, 1)),
    )
    monkeypatch.setattr(
        script,
        "_run_checkpoint_world_model",
        lambda **kwargs: (target_rollout, pred_rollout),
    )
    monkeypatch.setattr(
        script,
        "_tensor_video_to_frames",
        lambda video_btchw: [float(video_btchw[0, 0, 0, 0, 0].item())],
    )

    side_by_side_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _fake_build_side_by_side_video(*, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Record comparison inputs and return a sentinel tensor."""
        side_by_side_calls.append((left.clone(), right.clone()))
        return torch.full_like(right, 33.0)

    def _fake_export_video(*, video_frames: list[object], output_video_path: str, fps: int) -> None:
        """Capture export arguments without writing MP4 files."""
        export_calls.append(
            {
                "video_frames": video_frames,
                "output_video_path": output_video_path,
                "fps": fps,
            }
        )

    monkeypatch.setattr(script, "_build_side_by_side_video", _fake_build_side_by_side_video)
    monkeypatch.setattr(script, "_export_video", _fake_export_video)
    monkeypatch.setattr(
        script,
        "_write_plausibility_report",
        lambda **kwargs: plausibility_calls.append(kwargs) or {"plausible": True},
    )
    monkeypatch.setattr(
        script,
        "_write_arm_motion_report",
        lambda **kwargs: motion_calls.append(kwargs)
        or {
            "summary": {"motion_verdict": "good"},
            "artifacts": {"arm_crop_comparison_video": "crop.mp4"},
        },
    )

    result = script._run_one_checkpoint_resolution(
        mode="checkpoint",
        config_path=Path("configs/train/aloha_fork_pick_up.yaml"),
        checkpoint_path=Path("runs/example/checkpoints/step_0000100.pt"),
        width=320,
        height=240,
        output_path=tmp_path / "generated.mp4",
        comparison_path=tmp_path / "comparison.mp4",
        plausibility_output_path=tmp_path / "plausibility_report.json",
        motion_output_path=tmp_path / "arm_motion_report.json",
        repo_id="repo",
        episode_index=0,
        start_frame=0,
        video_key="observation.images.cam_high",
        context_len=9,
        horizon_len=8,
        k=1,
        integration_steps=10,
        fps=10,
        seed=0,
        single_chunk_rollout=True,
        device_name="cpu",
        action_source="auto",
        action_scale=1.0,
        action_token_scale=1.0,
        control_scale=None,
        prompt="",
        negative_prompt="",
        guidance_scale=1.0,
        max_sequence_length=128,
    )

    assert len(export_calls) == 2
    assert export_calls[0]["video_frames"] == [22.0]
    assert export_calls[0]["output_video_path"] == str(tmp_path / "generated.mp4")
    assert len(side_by_side_calls) == 1
    assert torch.equal(side_by_side_calls[0][0], target_rollout)
    assert torch.equal(side_by_side_calls[0][1], pred_rollout)
    assert export_calls[1]["video_frames"] == [33.0]
    assert export_calls[1]["output_video_path"] == str(tmp_path / "comparison.mp4")
    assert len(plausibility_calls) == 1
    assert torch.equal(plausibility_calls[0]["reference_video"], target_rollout)
    assert torch.equal(plausibility_calls[0]["generated_video"], pred_rollout)
    assert plausibility_calls[0]["output_json"] == tmp_path / "plausibility_report.json"
    assert len(motion_calls) == 1
    assert torch.equal(motion_calls[0]["reference_video"], target_rollout)
    assert torch.equal(motion_calls[0]["generated_video"], pred_rollout)
    assert motion_calls[0]["generated_video_path"] == tmp_path / "generated.mp4"
    assert motion_calls[0]["output_json"] == tmp_path / "arm_motion_report.json"
    assert result["plausibility_output_path"] == str(tmp_path / "plausibility_report.json")
    assert result["plausibility"] == {"plausible": True}
    assert result["motion_output_path"] == str(tmp_path / "arm_motion_report.json")
    assert result["motion"] == {
        "summary": {"motion_verdict": "good"},
        "artifacts": {"arm_crop_comparison_video": "crop.mp4"},
    }


def test_checkpoint_mode_control_scale_override_updates_runtime(monkeypatch, tmp_path) -> None:
    """Checkpoint sweeps should honor an explicit runtime control-scale override."""
    captured_control_scales: list[float] = []

    monkeypatch.setattr(
        script,
        "_load_checkpoint_runtime_config",
        lambda path: (
            {},
            SimpleNamespace(
                frame_width=320,
                frame_height=240,
                trainable_backbone="lora",
                conditioning_mode="action",
                context_len=9,
                horizon_len=8,
                control_scale=1.0,
            ),
        ),
    )
    monkeypatch.setattr(script, "_resolve_device", lambda device_name: torch.device("cpu"))
    monkeypatch.setattr(script, "_select_runtime_dtype", lambda device: torch.float32)
    monkeypatch.setattr(
        script,
        "_load_checkpoint_clip",
        lambda **kwargs: (torch.zeros(1, 17, 3, 240, 320), torch.zeros(17, 1)),
    )

    def _fake_run_checkpoint_world_model(*, runtime_cfg, **kwargs):
        """Capture the effective runtime control scale passed into checkpoint inference."""
        captured_control_scales.append(float(runtime_cfg.control_scale))
        target = torch.zeros(1, 2, 3, 4, 5)
        pred = torch.ones(1, 2, 3, 4, 5)
        return target, pred

    monkeypatch.setattr(script, "_run_checkpoint_world_model", _fake_run_checkpoint_world_model)
    monkeypatch.setattr(script, "_tensor_video_to_frames", lambda video_btchw: [0.0])
    monkeypatch.setattr(script, "_build_side_by_side_video", lambda **kwargs: torch.zeros(1, 2, 3, 4, 5))
    monkeypatch.setattr(script, "_export_video", lambda **kwargs: None)
    monkeypatch.setattr(script, "_write_plausibility_report", lambda **kwargs: {"plausible": True})
    monkeypatch.setattr(
        script,
        "_write_arm_motion_report",
        lambda **kwargs: {"summary": {"motion_verdict": "good"}},
    )

    result = script._run_one_checkpoint_resolution(
        mode="checkpoint",
        config_path=Path("configs/train/aloha_fork_pick_up.yaml"),
        checkpoint_path=Path("runs/example/checkpoints/step_0000100.pt"),
        width=320,
        height=240,
        output_path=tmp_path / "generated.mp4",
        comparison_path=tmp_path / "comparison.mp4",
        plausibility_output_path=tmp_path / "plausibility_report.json",
        motion_output_path=tmp_path / "arm_motion_report.json",
        repo_id="repo",
        episode_index=0,
        start_frame=0,
        video_key="observation.images.cam_high",
        context_len=9,
        horizon_len=8,
        k=1,
        integration_steps=10,
        fps=10,
        seed=0,
        single_chunk_rollout=True,
        device_name="cpu",
        action_source="auto",
        action_scale=1.0,
        action_token_scale=1.0,
        control_scale=1.5,
        prompt="",
        negative_prompt="",
        guidance_scale=1.0,
        max_sequence_length=128,
    )

    assert captured_control_scales == [1.5]
    assert result["conditioning_scale"] == 1.5
    assert "Overrode runtime control scale to 1.500 for this probe." in result["notes"]


def test_run_checkpoint_world_model_matches_current_action_infer_logic(monkeypatch) -> None:
    """Checkpoint sweeps should preserve current action-control, added-K/V, and residual-mode inference."""

    class _FakeActionEncoder(torch.nn.Module):
        """Return a fixed token tensor for checkpoint-mode action conditioning."""

        def forward(self, a_plan: torch.Tensor) -> torch.Tensor:
            """Project the action plan to a deterministic token tensor."""
            del a_plan
            return torch.full((1, 3, 4), 7.0)

    prepared = SimpleNamespace(
        a_plan=torch.arange(6, dtype=torch.float32).view(1, 3, 2),
        z_past_video=torch.full((1, 2, 2, 1, 1), 3.0),
        z_future_video=torch.full((1, 2, 3, 1, 1), 4.0),
    )
    captured_infer_kwargs: dict[str, object] = {}

    monkeypatch.setattr(script.WanVAE, "from_pretrained", lambda **kwargs: object())
    monkeypatch.setattr(script, "_infer_checkpoint_action_dim", lambda checkpoint: 2)
    monkeypatch.setattr(script, "_select_action_tensor", lambda **kwargs: torch.zeros(5, 2))
    monkeypatch.setattr(script, "prepare_packed_batch", lambda **kwargs: prepared)
    monkeypatch.setattr(
        script,
        "build_wan_vace_runtime_modules",
        lambda runtime_cfg, prepared_batch, device, checkpoint: (
            torch.nn.Identity(),
            _FakeActionEncoder(),
        ),
    )
    monkeypatch.setattr(
        script.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    def _fake_infer_future_videos_chunkwise(model, **kwargs):
        """Capture the inference kwargs and return a deterministic latent future."""
        del model
        captured_infer_kwargs.update(kwargs)
        return torch.full((1, 2, 3, 1, 1), 9.0)

    monkeypatch.setattr(script, "infer_future_videos_chunkwise", _fake_infer_future_videos_chunkwise)
    monkeypatch.setattr(
        script,
        "_decode_future_latents",
        lambda **kwargs: (
            torch.full((1, 3, 3, 1, 1), 8.0),
            torch.full((1, 3, 3, 1, 1), 6.0),
        ),
    )
    monkeypatch.setattr(
        script,
        "preprocess_video_for_vae",
        lambda video, frame_height, frame_width: torch.zeros(1, 5, 3, 1, 1),
    )
    monkeypatch.setattr(
        script,
        "_build_rollout_video",
        lambda *, target_full_video, pred_future_video, context_len: (
            torch.full((1, 5, 3, 1, 1), 8.0),
            torch.full((1, 5, 3, 1, 1), 6.0),
        ),
    )

    target_rollout, pred_rollout = script._run_checkpoint_world_model(
        runtime_cfg=SimpleNamespace(
            wan_vace_model_id="wan",
            conditioning_mode="action",
            context_len=2,
            horizon_len=3,
            chunk_schedule_mode="k_chunks",
            action_conditioning_window="chunk",
            action_backbone_added_kv_mode="reuse_action_tokens",
            future_latent_residual_mode="last_context_frame",
        ),
        checkpoint={},
        video=torch.zeros(1, 5, 3, 4, 4),
        action_seq=torch.zeros(5, 2),
        video_key="observation.images.cam_high",
        width=320,
        height=240,
        k=1,
        integration_steps=10,
        single_chunk_rollout=True,
        action_source="auto",
        device=torch.device("cpu"),
        runtime_dtype=torch.float32,
        generator=None,
    )

    assert torch.equal(target_rollout, torch.full((1, 5, 3, 1, 1), 8.0))
    assert torch.equal(pred_rollout, torch.full((1, 5, 3, 1, 1), 6.0))
    assert torch.equal(captured_infer_kwargs["image_attention_tokens"], torch.full((1, 3, 4), 7.0))
    assert captured_infer_kwargs["future_latent_residual_mode"] == "last_context_frame"


def test_run_checkpoint_world_model_supports_prompt_conditioning_without_checkpoint(monkeypatch) -> None:
    """Repo inference should pass prompt CFG tokens through chunkwise sampling without a checkpoint."""

    class _FakeBackbone(torch.nn.Module):
        """Backbone stub that exposes one parameter for dtype discovery."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    class _FakeModel(torch.nn.Module):
        """World-model stub with a backbone parameter and eval/to shims."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = _FakeBackbone()

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    class _UnusedActionEncoder(torch.nn.Module):
        """Action encoder stub that should not be touched in prompt mode."""

        def forward(self, a_plan: torch.Tensor) -> torch.Tensor:
            raise AssertionError("prompt mode should not use the action encoder")

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    prepared = SimpleNamespace(
        a_plan=torch.arange(6, dtype=torch.float32).view(1, 3, 2),
        z_past_video=torch.full((1, 2, 2, 1, 1), 3.0),
        z_future_video=torch.full((1, 2, 3, 1, 1), 4.0),
    )
    captured_infer_kwargs: dict[str, object] = {}

    monkeypatch.setattr(script.WanVAE, "from_pretrained", lambda **kwargs: object())
    monkeypatch.setattr(script, "prepare_packed_batch", lambda **kwargs: prepared)
    monkeypatch.setattr(
        script,
        "build_wan_vace_runtime_modules",
        lambda runtime_cfg, prepared_batch, device, checkpoint: (_FakeModel(), _UnusedActionEncoder()),
    )
    monkeypatch.setattr(
        script.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(script, "_load_prompt_encoder", lambda runtime_cfg: ("tok", "enc"))
    monkeypatch.setattr(
        script,
        "_build_prompt_conditioning_tokens",
        lambda **kwargs: (
            torch.full((1, 5, 4), 9.0, dtype=torch.float32),
            torch.full((1, 5, 4), 2.0, dtype=torch.float32),
        ),
    )

    def _fake_infer_future_videos_chunkwise(model, **kwargs):
        """Capture prompt-conditioning kwargs and return a deterministic latent future."""
        del model
        captured_infer_kwargs.update(kwargs)
        return torch.full((1, 2, 3, 1, 1), 9.0)

    monkeypatch.setattr(script, "infer_future_videos_chunkwise", _fake_infer_future_videos_chunkwise)
    monkeypatch.setattr(
        script,
        "_decode_future_latents",
        lambda **kwargs: (
            torch.full((1, 3, 3, 1, 1), 8.0),
            torch.full((1, 3, 3, 1, 1), 6.0),
        ),
    )
    monkeypatch.setattr(
        script,
        "preprocess_video_for_vae",
        lambda video, frame_height, frame_width: torch.zeros(1, 5, 3, 1, 1),
    )
    monkeypatch.setattr(
        script,
        "_build_rollout_video",
        lambda *, target_full_video, pred_future_video, context_len: (
            torch.full((1, 5, 3, 1, 1), 8.0),
            torch.full((1, 5, 3, 1, 1), 6.0),
        ),
    )

    target_rollout, pred_rollout = script._run_checkpoint_world_model(
        runtime_cfg=SimpleNamespace(
            wan_vace_model_id="wan",
            conditioning_mode="prompt",
            prompt="pick up the fork",
            negative_prompt="distorted colors",
            guidance_scale=5.0,
            max_sequence_length=128,
            context_len=2,
            horizon_len=3,
            chunk_schedule_mode="k_chunks",
            action_conditioning_window="chunk",
            future_latent_residual_mode="none",
        ),
        checkpoint=None,
        video=torch.zeros(1, 5, 3, 4, 4),
        action_seq=torch.zeros(1, 5, 2),
        video_key="observation.images.cam_high",
        width=320,
        height=240,
        k=1,
        integration_steps=10,
        single_chunk_rollout=True,
        action_source="auto",
        device=torch.device("cpu"),
        runtime_dtype=torch.float32,
        generator=None,
    )

    assert torch.equal(target_rollout, torch.full((1, 5, 3, 1, 1), 8.0))
    assert torch.equal(pred_rollout, torch.full((1, 5, 3, 1, 1), 6.0))
    assert torch.equal(captured_infer_kwargs["cross_attention_tokens"], torch.full((1, 5, 4), 9.0))
    assert torch.equal(captured_infer_kwargs["negative_cross_attention_tokens"], torch.full((1, 5, 4), 2.0))
    assert captured_infer_kwargs["guidance_scale"] == 5.0
    assert captured_infer_kwargs["chunk_conditioning"] is False
    assert captured_infer_kwargs["image_attention_tokens"] is None
