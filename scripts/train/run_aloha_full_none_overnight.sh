#!/usr/bin/env bash
# Cache the full ALOHA fork-pick-up dataset, then launch staged full-dataset training.

set -euo pipefail

cd /home/amaniscalco/world-model-v1
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

validate_stage() {
  local output_dir="$1"
  local expected_step="$2"

  python scripts/check/validate_training_stage.py \
    --output-dir "$output_dir" \
    --expected-step "$expected_step"
}

resolve_codex_bin() {
  local detected_path
  local matches

  if [[ -n "${CODEX_BIN:-}" && -x "${CODEX_BIN}" ]]; then
    printf '%s\n' "${CODEX_BIN}"
    return 0
  fi

  detected_path="$(command -v codex || true)"
  if [[ -n "$detected_path" && -x "$detected_path" ]]; then
    printf '%s\n' "$detected_path"
    return 0
  fi

  matches=(/home/amaniscalco/.antigravity/extensions/openai.chatgpt-*/bin/*/codex)
  if [[ -e "${matches[0]}" && -x "${matches[0]}" ]]; then
    printf '%s\n' "${matches[0]}"
    return 0
  fi

  return 1
}

codex_stage_check() {
  local output_dir="$1"
  local expected_step="$2"
  local checkpoint_path
  local report_path
  local codex_bin

  if [[ "${RUN_CODEX_STAGE_CHECK:-0}" != "1" ]]; then
    return 0
  fi

  codex_bin="$(resolve_codex_bin || true)"
  if [[ -z "$codex_bin" ]]; then
    echo "RUN_CODEX_STAGE_CHECK=1 was set, but codex is not installed." >&2
    return 1
  fi

  checkpoint_path="${output_dir}/checkpoints/step_$(printf '%07d' "$expected_step").pt"
  report_path="$(mktemp)"
  "$codex_bin" exec \
    --cd /home/amaniscalco/world-model-v1 \
    --skip-git-repo-check \
    --output-last-message "$report_path" \
    "Inspect ${output_dir}/metrics.jsonl and ${checkpoint_path}. Reply with exactly PASS or FAIL on the first line. Return FAIL if the checkpoint is missing, the last logged step is not ${expected_step}, or the final loss is non-finite." \
    >/dev/null
  grep -qx 'PASS' "$report_path"
  rm -f "$report_path"
}

run_stage() {
  local label="$1"
  local output_dir="$2"
  local max_steps="$3"
  local resume_from="${4:-}"

  if validate_stage "$output_dir" "$max_steps" >/dev/null 2>&1; then
    echo "== ${label} already validated at step ${max_steps}; skipping training =="
    codex_stage_check "$output_dir" "$max_steps"
    return 0
  fi

  echo "== ${label} =="
  if [[ -n "$resume_from" ]]; then
    python scripts/train/world_model.py \
      --config configs/train/aloha_fork_pick_up.yaml \
      --conditioning-mode none \
      --auto-stop-check-every 0 \
      --resume-from "$resume_from" \
      --output-dir "$output_dir" \
      --max-steps "$max_steps"
  else
    python scripts/train/world_model.py \
      --config configs/train/aloha_fork_pick_up.yaml \
      --conditioning-mode none \
      --auto-stop-check-every 0 \
      --output-dir "$output_dir" \
      --max-steps "$max_steps"
  fi

  validate_stage "$output_dir" "$max_steps"
  codex_stage_check "$output_dir" "$max_steps"
}

echo "== Caching full lerobot/aloha_static_fork_pick_up dataset =="
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "lerobot/aloha_static_fork_pick_up",
    episodes=None,
    video_backend="pyav",
)
print(f"Cached dataset frames={len(dataset)} episodes={dataset.meta.total_episodes}")
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_stage \
  "Multi-episode LoRA none run: 0 -> 200" \
  "runs/test_full_multi_320x240_lora8_none" \
  "200"

run_stage \
  "Resume multi-episode LoRA none run: 200 -> 400" \
  "runs/test_full_multi_320x240_lora8_none" \
  "400" \
  "runs/test_full_multi_320x240_lora8_none/checkpoints/step_0000200.pt"

run_stage \
  "Resume multi-episode LoRA none run: 400 -> 800" \
  "runs/test_full_multi_320x240_lora8_none" \
  "800" \
  "runs/test_full_multi_320x240_lora8_none/checkpoints/step_0000400.pt"
