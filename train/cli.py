"""Command-line orchestration for LoRA DPO training (paper §2.6, Appendix E)."""

from __future__ import annotations

import train.gpu_env  # noqa: F401 — set BNB_CUDA_VERSION before HF/bnb import

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from train.paths import infer_variant, write_latest_pointer
from train.solver import load_pairs
from train.counterfactual import (
    find_resume_for_publish,
    resolve_checkpoint_path,
    resolve_run_dirs,
)
from train.frontier import build_dpo_trainer, extract_train_metrics, prepare_model
from runs.paths import ensure_session, lora_variant_dir
from train.run_manifest import (
    build_config_snapshot,
    finalize_run_manifest,
    persist_run_manifest,
    publish_adapter,
)
from train.utils import build_lora_config


def _resolve_train_args(args: argparse.Namespace) -> None:
    if args.paper:
        args.model_id = args.model_id or config.PAPER_MODEL_ID
        args.lora_r = config.PAPER_LORA_R if args.lora_r is None else args.lora_r
        args.lora_alpha = config.PAPER_LORA_ALPHA if args.lora_alpha is None else args.lora_alpha
        args.lora_target = args.lora_target or config.PAPER_LORA_TARGET_MODULES
        if args.out in ("lora/all", "lora_3b/all"):
            args.out = "all"
    if args.max_steps is None:
        args.max_steps = config.max_train_steps_for_variant(infer_variant(Path(args.out)))
    if args.max_length is None:
        args.max_length = config.MAX_SEQ_LENGTH if args.paper else config.LOCAL_MAX_SEQ_LENGTH
    if args.grad_accum is None:
        args.grad_accum = config.GRADIENT_ACCUMULATION if args.paper else config.LOCAL_GRADIENT_ACCUMULATION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=None, help="JSONL with prompt/chosen/rejected")
    parser.add_argument("--out", default="lora/all", help="Variant publish dir (e.g. lora/all)")
    parser.add_argument("--model-id", default=None, help="Base HF model (default: config.MODEL_ID)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Stop after N optimizer steps (overrides --epochs when set). "
            "Defaults to TRAIN_MAX_STEPS_<VARIANT>, then TRAIN_MAX_STEPS."
        ),
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
        resume_triple = resolve_checkpoint_path(args.resume_from_checkpoint)
    elif args.resume:
        resume_triple = find_resume_for_publish(publish_dir)
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

    run_dir, adapter_dir, publish_dir = resolve_run_dirs(
        args, publish_dir=publish_dir, resume=resume_triple
    )
    resume_ckpt = resume_triple[2] if resume_triple else None
    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute():
        pairs_path = config.PROJECT_ROOT / pairs_path

    model, tokenizer = prepare_model(
        model_id=model_id,
        lora_config=lora_cfg,
        resume_checkpoint=resume_ckpt,
    )

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
    is_main_process = int(os.environ.get("RANK", "0")) == 0
    if resume_ckpt is None and is_main_process:
        persist_run_manifest(run_dir, argv=sys.argv, args=args, snapshot=snapshot)
    elif resume_ckpt is not None and is_main_process:
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
    trainer = build_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        args=args,
        adapter_dir=adapter_dir,
        tensorboard_dir=tb_dir,
        use_tensorboard=use_tensorboard,
        save_total_limit=save_total_limit,
    )
    status = "completed"
    train_metrics: dict | None = None
    try:
        trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
        trainer.save_model(str(adapter_dir))
        train_metrics = extract_train_metrics(trainer)
    except Exception:
        status = "failed"
        if is_main_process:
            finalize_run_manifest(run_dir, status=status)
        raise
    else:
        if is_main_process:
            publish_adapter(adapter_dir, publish_dir)
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
