"""Run allocation, checkpoint cleanup, and resume discovery."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from transformers import TrainerCallback

import config
from train.paths import infer_variant
from runs.paths import ensure_session, new_lora_variant_dir

_CHECKPOINT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
}


def _is_checkpoint_file(path: Path) -> bool:
    """Keep resume-critical files, including one RNG state per DDP rank."""
    return path.name in _CHECKPOINT_FILES or (
        path.name.startswith("rng_state_") and path.suffix == ".pth"
    )


class MinimalCheckpointCallback(TrainerCallback):
    """Trim resumable checkpoints and retain the lowest-loss checkpoint."""

    def on_save(self, args, state, control, **kwargs):
        if not args.should_save:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.is_dir():
            return control
        for path in checkpoint_dir.iterdir():
            if _is_checkpoint_file(path):
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        losses = [
            entry["loss"]
            for entry in state.log_history
            if entry.get("step") == state.global_step and isinstance(entry.get("loss"), (int, float))
        ]
        if not losses:
            return control
        run_dir = Path(args.output_dir).parent
        info_path = run_dir / "run_info.json"
        manifest = {}
        if info_path.is_file():
            manifest = json.loads(info_path.read_text(encoding="utf-8"))
        current_loss = float(losses[-1])
        best_loss = manifest.get("best_train_loss")
        if best_loss is None or current_loss < float(best_loss):
            best_dir = run_dir / "best_checkpoint"
            staging_dir = run_dir / ".best_checkpoint.tmp"
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            shutil.copytree(checkpoint_dir, staging_dir)
            if best_dir.exists():
                shutil.rmtree(best_dir)
            staging_dir.rename(best_dir)
            manifest["best_train_loss"] = current_loss
            manifest["best_checkpoint_step"] = state.global_step
            manifest["best_checkpoint"] = str(best_dir.resolve())
            info_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return control


def _latest_checkpoint(adapter_dir: Path) -> Path | None:
    checkpoints = sorted(
        adapter_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def _publish_dir_matches(manifest: dict, publish_dir: Path) -> bool:
    variant = infer_variant(publish_dir)
    snapshot = manifest.get("resolved_config") or {}
    if snapshot.get("variant") == variant:
        return True
    published = manifest.get("publish_dir") or snapshot.get("publish_dir") or ""
    try:
        return Path(published).resolve() == publish_dir.resolve()
    except OSError:
        return str(published).endswith(f"/{variant}") or str(published).endswith(variant)


def find_resume_for_publish(publish_dir: Path) -> tuple[Path, Path, Path] | None:
    """Return (run_dir, adapter_dir, checkpoint_dir) for the latest unfinished run."""
    best: tuple[float, Path, Path, Path] | None = None
    if not config.RUNS_DIR.is_dir():
        return None
    session_filter = os.environ.get("RUN_ID")
    sessions = (
        [config.RUNS_DIR / session_filter]
        if session_filter
        else sorted(path for path in config.RUNS_DIR.iterdir() if path.is_dir())
    )
    variant = infer_variant(publish_dir)
    for session in sessions:
        run_dir = session / "lora" / variant
        info_path = run_dir / "run_info.json"
        adapter_dir = run_dir / "adapter"
        if not info_path.is_file() or not adapter_dir.is_dir():
            continue
        manifest = json.loads(info_path.read_text(encoding="utf-8"))
        checkpoint = _latest_checkpoint(adapter_dir)
        if (
            not _publish_dir_matches(manifest, publish_dir)
            or checkpoint is None
            or manifest.get("status") == "completed"
        ):
            continue
        candidate = (checkpoint.stat().st_mtime, run_dir, adapter_dir, checkpoint)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return None if best is None else best[1:]


def resolve_checkpoint_path(path: Path) -> tuple[Path, Path, Path]:
    """Resolve a supplied path to (run_dir, adapter_dir, checkpoint_dir)."""
    path = path.resolve()
    if path.name.startswith("checkpoint-"):
        return path.parent.parent, path.parent, path
    if (path / "trainer_state.json").is_file():
        return path.parent.parent, path.parent, path
    if path.is_dir() and (path / "adapter_config.json").is_file():
        checkpoint = _latest_checkpoint(path)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint-* under {path}")
        return path.parent, path, checkpoint
    raise FileNotFoundError(f"Not a checkpoint path: {path}")


def resolve_run_dirs(
    args,
    *,
    publish_dir: Path,
    resume: tuple[Path, Path, Path] | None,
) -> tuple[Path, Path, Path]:
    """Return (run_dir, adapter_dir, publish_dir)."""
    if resume is not None:
        run_dir, adapter_dir, _checkpoint = resume
        return run_dir, adapter_dir, run_dir
    if args.no_timestamp_out:
        publish_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = publish_dir / "adapter"
        if not adapter_dir.is_dir():
            adapter_dir = publish_dir
        return publish_dir, adapter_dir, publish_dir
    ensure_session()
    run_dir = new_lora_variant_dir(infer_variant(publish_dir))
    return run_dir, run_dir / "adapter", run_dir
