"""Train LoRA with DPO on preference pairs (paper §2.6, Appendix E)."""

from __future__ import annotations

import lora.gpu_env  # noqa: F401 — set BNB_CUDA_VERSION before HF/bnb import

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from transformers import TrainerCallback

import config
from lora.paths import infer_variant, write_latest_pointer
from runs.paths import ensure_session, lora_variant_dir, new_lora_variant_dir
from lora.run_manifest import (
    build_config_snapshot,
    finalize_run_manifest,
    persist_run_manifest,
    publish_adapter,
)
from lora.utils import attach_lora, build_lora_config, load_base_model, load_lora_adapter


_CHECKPOINT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
}


class MinimalCheckpointCallback(TrainerCallback):
    """Keep only the adapter and Trainer state required to resume a run."""

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.is_dir():
            return control
        for path in checkpoint_dir.iterdir():
            if path.name in _CHECKPOINT_FILES:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        return control


def _latest_checkpoint(adapter_dir: Path) -> Path | None:
    cps = sorted(
        adapter_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    return cps[-1] if cps else None


def _publish_dir_matches(manifest: dict, publish_dir: Path) -> bool:
    variant = infer_variant(publish_dir)
    snap = manifest.get("resolved_config") or {}
    if snap.get("variant") == variant:
        return True
    pub = manifest.get("publish_dir") or snap.get("publish_dir") or ""
    try:
        return Path(pub).resolve() == publish_dir.resolve()
    except OSError:
        return str(pub).endswith(f"/{variant}") or str(pub).endswith(variant)


def _find_resume_for_publish(publish_dir: Path) -> tuple[Path, Path, Path] | None:
    """Return (run_dir, adapter_dir, checkpoint_dir) for latest in-progress run."""
    best: tuple[float, Path, Path, Path] | None = None
    variant = infer_variant(publish_dir)
    runs_dir = config.RUNS_DIR
    if not runs_dir.is_dir():
        return None

    session_filter = os.environ.get("RUN_ID")
    if session_filter:
        sessions = [runs_dir / session_filter]
    else:
        sessions = sorted(p for p in runs_dir.iterdir() if p.is_dir())

    for session in sessions:
        run_dir = session / "lora" / variant
        info_path = run_dir / "run_info.json"
        adapter_dir = run_dir / "adapter"
        if not info_path.is_file() or not adapter_dir.is_dir():
            continue
        manifest = json.loads(info_path.read_text(encoding="utf-8"))
        if not _publish_dir_matches(manifest, publish_dir):
            continue
        ckpt = _latest_checkpoint(adapter_dir)
        if ckpt is None:
            continue
        if manifest.get("status") == "completed":
            continue
        score = ckpt.stat().st_mtime
        if best is None or score > best[0]:
            best = (score, run_dir, adapter_dir, ckpt)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _resolve_checkpoint_path(path: Path) -> tuple[Path, Path, Path]:
    """Return (run_dir, adapter_dir, checkpoint_dir)."""
    path = path.resolve()
    if path.name.startswith("checkpoint-"):
        adapter_dir = path.parent
        return adapter_dir.parent, adapter_dir, path
    if (path / "trainer_state.json").is_file():
        return path.parent.parent, path.parent, path
    if path.is_dir() and (path / "adapter_config.json").is_file():
        ckpt = _latest_checkpoint(path)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint-* under {path}")
        return path.parent, path, ckpt
    raise FileNotFoundError(f"Not a checkpoint path: {path}")


def load_pairs(path: Path) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(
                {
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                }
            )
    if not rows:
        raise ValueError(f"No pairs in {path}")
    return Dataset.from_list(rows)


def _resolve_train_args(args: argparse.Namespace) -> None:
    if args.paper:
        args.model_id = args.model_id or config.PAPER_MODEL_ID
        args.lora_r = config.PAPER_LORA_R if args.lora_r is None else args.lora_r
        args.lora_alpha = config.PAPER_LORA_ALPHA if args.lora_alpha is None else args.lora_alpha
        args.lora_target = args.lora_target or config.PAPER_LORA_TARGET_MODULES
        if args.out in ("lora/all", "lora_3b/all"):
            args.out = "all"
    if args.max_length is None:
        args.max_length = config.MAX_SEQ_LENGTH if args.paper else config.LOCAL_MAX_SEQ_LENGTH
    if args.grad_accum is None:
        args.grad_accum = config.GRADIENT_ACCUMULATION if args.paper else config.LOCAL_GRADIENT_ACCUMULATION


def _resolve_run_dirs(
    args: argparse.Namespace,
    *,
    publish_dir: Path,
    resume: tuple[Path, Path, Path] | None,
) -> tuple[Path, Path, Path]:
    """Return (run_dir, adapter_dir, publish_dir)."""
    if resume is not None:
        run_dir, adapter_dir, _ckpt = resume
        return run_dir, adapter_dir, run_dir
    if args.no_timestamp_out:
        publish_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = publish_dir / "adapter"
        if not adapter_dir.is_dir():
            adapter_dir = publish_dir
        return publish_dir, adapter_dir, publish_dir
    variant = infer_variant(publish_dir)
    ensure_session()
    run_dir = new_lora_variant_dir(variant)
    adapter_dir = run_dir / "adapter"
    return run_dir, adapter_dir, run_dir


def _extract_train_metrics(trainer: DPOTrainer) -> dict:
    state = trainer.state
    metrics: dict = {
        "global_step": state.global_step,
        "epoch": state.epoch,
    }
    if state.log_history:
        last = state.log_history[-1]
        for key in ("train_runtime", "train_loss", "train_samples_per_second", "train_steps_per_second"):
            if key in last:
                metrics[key] = last[key]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=None, help="JSONL with prompt/chosen/rejected")
    parser.add_argument("--out", default="lora/all", help="Variant publish dir (e.g. lora/all)")
    parser.add_argument("--model-id", default=None, help="Base HF model (default: config.MODEL_ID)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=config.MAX_TRAIN_STEPS,
        help="Stop after N optimizer steps (overrides --epochs when set)",
    )
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=config.DPO_BETA)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument(
        "--lora-target",
        default=None,
        help="LoRA target modules, e.g. q_proj,v_proj or all-linear",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Use Qwen2.5-3B + paper LoRA r=16 alpha=32 all-linear",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=f"Max tokens per DPO example (default: {config.LOCAL_MAX_SEQ_LENGTH} local, "
        f"{config.MAX_SEQ_LENGTH} with --paper).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help=f"Gradient accumulation steps (default: {config.LOCAL_GRADIENT_ACCUMULATION} local, "
        f"{config.GRADIENT_ACCUMULATION} with --paper).",
    )
    parser.add_argument(
        "--no-timestamp-out",
        action="store_true",
        help="Write directly to --out (legacy). Default: runs/<session>/lora/<variant>/",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Log train loss to TensorBoard under <run_dir>/tensorboard/ (default with --paper)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard even with --paper",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help=f"Max checkpoints to keep (default: {config.PAPER_SAVE_TOTAL_LIMIT} paper, 2 local)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume latest unfinished checkpoint for --out variant (under runs/<session>/lora/)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume from explicit checkpoint dir (e.g. runs/<id>/lora/all/adapter/checkpoint-500)",
    )
    args = parser.parse_args()
    _resolve_train_args(args)
    if args.resume and args.resume_from_checkpoint:
        parser.error("Use only one of --resume or --resume-from-checkpoint")

    publish_dir = Path(args.out)
    if not publish_dir.is_absolute():
        publish_dir = config.PROJECT_ROOT / publish_dir
    variant_name = infer_variant(publish_dir)
    if not args.no_timestamp_out:
        ensure_session()
        publish_dir = lora_variant_dir(variant_name)

    resume_triple: tuple[Path, Path, Path] | None = None
    if args.resume_from_checkpoint is not None:
        resume_triple = _resolve_checkpoint_path(args.resume_from_checkpoint)
    elif args.resume:
        resume_triple = _find_resume_for_publish(publish_dir)
        if resume_triple is None:
            parser.error(f"--resume: no unfinished checkpoint found for {publish_dir}")

    use_tensorboard = (args.tensorboard or args.paper) and not args.no_tensorboard
    save_total_limit = (
        args.save_total_limit
        if args.save_total_limit is not None
        else (config.PAPER_SAVE_TOTAL_LIMIT if args.paper else 2)
    )

    model_id = args.model_id or config.MODEL_ID
    lora_r = args.lora_r if args.lora_r is not None else config.LORA_R
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else config.LORA_ALPHA
    lora_target = args.lora_target or ",".join(config.LORA_TARGET_MODULES)
    target_modules = (
        lora_target if lora_target == "all-linear" else lora_target.split(",")
    )
    lora_cfg = build_lora_config(r=lora_r, alpha=lora_alpha, target_modules=target_modules)

    if not args.pairs:
        parser.error("--pairs is required")

    run_dir, adapter_dir, publish_dir = _resolve_run_dirs(
        args, publish_dir=publish_dir, resume=resume_triple
    )
    resume_ckpt = resume_triple[2] if resume_triple else None
    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute():
        pairs_path = config.PROJECT_ROOT / pairs_path

    if resume_ckpt is not None:
        model, tokenizer = load_lora_adapter(
            resume_ckpt,
            model_id=model_id,
            use_4bit=config.USE_4BIT,
        )
        print(f"Resuming from {resume_ckpt}", flush=True)
    else:
        model, tokenizer = load_base_model(model_id=model_id, use_4bit=config.USE_4BIT)
        model = attach_lora(model, lora_cfg)
    if config.GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    model.print_trainable_parameters()

    dataset = load_pairs(pairs_path)
    variant = infer_variant(publish_dir)
    snapshot = build_config_snapshot(
        variant=variant,
        pairs_path=pairs_path,
        pairs_count=len(dataset),
        model_id=model_id,
        paper=args.paper,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target=lora_target,
        max_length=args.max_length,
        grad_accum=args.grad_accum,
        publish_dir=publish_dir,
    )
    if resume_ckpt is None:
        persist_run_manifest(run_dir, argv=sys.argv, args=args, snapshot=snapshot)
    else:
        info = run_dir / "run_info.json"
        if info.is_file():
            manifest = json.loads(info.read_text(encoding="utf-8"))
            manifest["status"] = "running"
            manifest["resumed_from"] = str(
                resume_ckpt.relative_to(config.PROJECT_ROOT)
                if resume_ckpt.is_relative_to(config.PROJECT_ROOT)
                else resume_ckpt
            )
            manifest["resume_command"] = " ".join(sys.argv)
            info.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"DPO train model={model_id} pairs={len(dataset)} "
        f"epochs={args.epochs} max_steps={args.max_steps} "
        f"max_length={args.max_length} grad_accum={args.grad_accum} "
        f"run_dir={run_dir} publish={publish_dir} "
        f"tensorboard={use_tensorboard} save_total_limit={save_total_limit} "
        f"resume={resume_ckpt}",
        flush=True,
    )
    tb_dir = run_dir / "tensorboard"
    train_kw: dict = {
        "output_dir": str(adapter_dir),
        "per_device_train_batch_size": config.PER_DEVICE_BATCH,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
    }
    if args.max_steps is not None:
        train_kw["max_steps"] = args.max_steps
    dpo_args = DPOConfig(
        **train_kw,
        learning_rate=args.lr,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=config.WARMUP_RATIO,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        save_total_limit=save_total_limit,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        # fp16 GradScaler breaks on torch 2.11+win (bf16 unscale NotImplementedError)
        fp16=config.TRAIN_FP16,
        bf16=config.TRAIN_BF16,
        report_to="tensorboard" if use_tensorboard else "none",
        logging_dir=str(tb_dir) if use_tensorboard else None,
        beta=args.beta,
        max_length=args.max_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[MinimalCheckpointCallback()],
    )
    status = "completed"
    train_metrics: dict | None = None
    try:
        trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
        trainer.save_model(str(adapter_dir))
        publish_adapter(adapter_dir, publish_dir)
        train_metrics = _extract_train_metrics(trainer)
    except Exception:
        status = "failed"
        finalize_run_manifest(run_dir, status=status)
        raise
    else:
        finalize_run_manifest(run_dir, status=status, train_metrics=train_metrics)
        if not args.no_timestamp_out:
            write_latest_pointer(run_dir)

    print(
        f"Saved LoRA adapter run={run_dir} publish={publish_dir} (base={model_id})",
        flush=True,
    )
    if use_tensorboard:
        print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}", flush=True)
    print(
        f"Checkpoints: {adapter_dir}/checkpoint-* (every {config.SAVE_STEPS} steps, "
        f"keep last {save_total_limit})",
        flush=True,
    )


if __name__ == "__main__":
    main()
