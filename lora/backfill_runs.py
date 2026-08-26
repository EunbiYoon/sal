"""Backfill lora/runs/<id>/ snapshots for adapters trained before timestamped runs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from lora.paths import write_latest_pointer
from lora.run_manifest import (
    build_config_snapshot,
    finalize_run_manifest,
    persist_run_manifest,
    publish_adapter,
    utc_now_iso,
)


def _guess_pairs_path(variant_dir: Path) -> Path | None:
    name = variant_dir.name
    candidates = [
        config.DPO_DATA_DIR / name / "a_beta_all.jsonl",
        config.DPO_DATA_DIR / "a_beta_all.jsonl",
    ]
    if name == "all":
        candidates = [
            config.DPO_DATA_DIR / "negotiation_1000" / "a_beta_all.jsonl",
            config.DPO_DATA_DIR / "a_beta_all.jsonl",
        ] + candidates
    for path in candidates:
        if path.is_file():
            return path
    return None


def _count_pairs(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _latest_checkpoint(adapter_root: Path) -> Path | None:
    cps = sorted(adapter_root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return cps[-1] if cps else None


def _train_metrics_from_state(adapter_root: Path) -> dict | None:
    ckpt = _latest_checkpoint(adapter_root)
    state_path = (ckpt or adapter_root) / "trainer_state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    metrics = {"global_step": state.get("global_step"), "epoch": state.get("epoch")}
    history = state.get("log_history") or []
    if history:
        last = history[-1]
        for key in ("train_runtime", "train_loss", "train_samples_per_second", "train_steps_per_second"):
            if key in last:
                metrics[key] = last[key]
    return metrics


def _mtime_stamp(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def backfill_variant(
    publish_dir: Path,
    *,
    pairs_path: Path | None,
    run_id: str | None,
    command: str | None,
) -> Path:
    if not (publish_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No adapter in {publish_dir}")

    pairs_path = pairs_path or _guess_pairs_path(publish_dir)
    if pairs_path is None:
        raise FileNotFoundError(f"Could not infer --pairs for {publish_dir}")
    if not pairs_path.is_absolute():
        pairs_path = config.PROJECT_ROOT / pairs_path

    pairs_count = _count_pairs(pairs_path)
    variant = publish_dir.name
    cfg = json.loads((publish_dir / "adapter_config.json").read_text(encoding="utf-8"))
    model_id = cfg.get("base_model_name_or_path", config.MODEL_ID)

    stamp = run_id or f"{_mtime_stamp(publish_dir / 'adapter_config.json')}_retro_{variant}"
    run_dir = config.LORA_RUNS_DIR / stamp
    adapter_dir = run_dir / "adapter"
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")

    adapter_dir.mkdir(parents=True)
    for item in publish_dir.iterdir():
        dest = adapter_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    ckpt = _latest_checkpoint(publish_dir)
    if ckpt is not None and not (adapter_dir / ckpt.name).exists():
        shutil.copytree(ckpt, adapter_dir / ckpt.name)

    snapshot = build_config_snapshot(
        variant=variant,
        pairs_path=pairs_path,
        pairs_count=pairs_count,
        model_id=model_id,
        paper=False,
        epochs=1,
        lr=config.LEARNING_RATE,
        beta=config.DPO_BETA,
        lora_r=cfg.get("r", config.LORA_R),
        lora_alpha=cfg.get("lora_alpha", config.LORA_ALPHA),
        lora_target=",".join(cfg.get("target_modules") or config.LORA_TARGET_MODULES),
        max_length=config.LOCAL_MAX_SEQ_LENGTH,
        grad_accum=config.LOCAL_GRADIENT_ACCUMULATION,
        publish_dir=publish_dir,
    )

    class _Args:
        no_timestamp_out = False

    argv = (
        command.split()
        if command
        else [
            sys.executable,
            "-m",
            "lora.train",
            "--pairs",
            str(pairs_path.relative_to(config.PROJECT_ROOT)),
            "--out",
            str(publish_dir.relative_to(config.PROJECT_ROOT)),
            "--epochs",
            "1",
        ]
    )
    persist_run_manifest(run_dir, argv=argv, args=_Args(), snapshot=snapshot)
    metrics = _train_metrics_from_state(publish_dir)
    manifest_path = run_dir / "run_info.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    manifest["finished_at"] = utc_now_iso()
    manifest["backfilled"] = True
    manifest["train_metrics"] = metrics
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    finalize_run_manifest(run_dir, status="completed", train_metrics=metrics)
    publish_adapter(adapter_dir, publish_dir)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill lora/runs snapshots for existing adapters")
    parser.add_argument("--publish-dir", type=Path, action="append", default=[], help="e.g. lora/all")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--command", default=None, help="Original shell command (for README)")
    parser.add_argument("--all-known", action="store_true", help="Backfill lora/all and lora/bos_1000")
    args = parser.parse_args()

    targets: list[Path] = list(args.publish_dir)
    if args.all_known:
        targets.extend([config.LORA_DIR / "all", config.LORA_DIR / "bos_1000"])

    if not targets:
        parser.error("Specify --publish-dir and/or --all-known")

    last_run: Path | None = None
    for publish_dir in targets:
        publish_dir = publish_dir if publish_dir.is_absolute() else config.PROJECT_ROOT / publish_dir
        pairs = args.pairs if args.pairs and len(targets) == 1 else None
        run_dir = backfill_variant(
            publish_dir,
            pairs_path=pairs,
            run_id=args.run_id if len(targets) == 1 else None,
            command=args.command if len(targets) == 1 else None,
        )
        print(f"Backfilled {publish_dir} -> {run_dir}")
        last_run = run_dir

    if last_run is not None:
        write_latest_pointer(last_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
