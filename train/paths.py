"""LoRA train output paths under runs/<session>/lora/<variant>/."""

from __future__ import annotations

from pathlib import Path

from runs.paths import new_lora_variant_dir, write_latest_pointer as write_session_latest


def infer_variant(publish_dir: Path) -> str:
    """e.g. runs/<id>/lora/all -> all, lora_3b/all -> all."""
    return publish_dir.name


def write_latest_pointer(run_dir: Path) -> None:
    session_id = run_dir.parent.parent.name
    write_session_latest(session_id)
