"""Shared path helpers for the training-optimizer workflow."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MEMORY_RELATIVE_PATH = Path("docs/investigation.md")
DEFAULT_PROMPT_RELATIVE_PATH = Path("docs/controller_prompt_investigations.md")
DEFAULT_STATE_RELATIVE_PATH = Path("runs/training_optimizer/investigation_controller_state.json")
TRAINING_OPTIMIZER_RUN_RELATIVE_ROOT = Path("runs/training_optimizer")
CONTROLLER_LOGS_RELATIVE_ROOT = TRAINING_OPTIMIZER_RUN_RELATIVE_ROOT / "controller_logs"
CODEX_DEBUG_RELATIVE_ROOT = TRAINING_OPTIMIZER_RUN_RELATIVE_ROOT / "debug"


def default_memory_path(repo_root: Path = REPO_ROOT) -> Path:
    """Return the default mutable investigation-memory markdown path."""
    return repo_root / DEFAULT_MEMORY_RELATIVE_PATH


def default_prompt_path(repo_root: Path = REPO_ROOT) -> Path:
    """Return the default static controller-prompt markdown path."""
    return repo_root / DEFAULT_PROMPT_RELATIVE_PATH


def default_state_path(repo_root: Path = REPO_ROOT) -> Path:
    """Return the default shared-session controller state JSON path."""
    return repo_root / DEFAULT_STATE_RELATIVE_PATH


def training_optimizer_run_root(repo_root: Path = REPO_ROOT) -> Path:
    """Return the root directory for optimizer-generated run artifacts."""
    return repo_root / TRAINING_OPTIMIZER_RUN_RELATIVE_ROOT


def controller_logs_root(repo_root: Path = REPO_ROOT) -> Path:
    """Return the directory where controller stdout/stderr logs are stored."""
    return repo_root / CONTROLLER_LOGS_RELATIVE_ROOT


def codex_debug_root(repo_root: Path = REPO_ROOT) -> Path:
    """Return the directory where Codex debug artifacts are written."""
    return repo_root / CODEX_DEBUG_RELATIVE_ROOT


def resolve_repo_relative_path(
    raw_path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Resolve an absolute or repo-relative path against `repo_root`."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def display_repo_path(path: str | Path, *, repo_root: Path = REPO_ROOT) -> str:
    """Render a path relative to `repo_root` when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def derive_state_path_for_memory_path(
    memory_path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Map one memory markdown path to its matching controller state JSON path."""
    resolved_memory_path = resolve_repo_relative_path(memory_path, repo_root=repo_root)
    if resolved_memory_path == default_memory_path(repo_root):
        return default_state_path(repo_root)

    try:
        relative_memory_path = resolved_memory_path.relative_to(repo_root)
    except ValueError:
        relative_memory_path = Path(resolved_memory_path.name)

    stem_parts = list(relative_memory_path.with_suffix("").parts)
    if stem_parts and stem_parts[0] == "docs":
        stem_parts = stem_parts[1:]
    state_stem = "_".join(part for part in stem_parts if part) or resolved_memory_path.stem
    return training_optimizer_run_root(repo_root) / f"{state_stem}_controller_state.json"
