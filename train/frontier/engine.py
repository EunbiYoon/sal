"""Model preparation and DPO trainer assembly."""

from __future__ import annotations

from pathlib import Path

from trl import DPOConfig, DPOTrainer

import config
from train.counterfactual import MinimalCheckpointCallback
from train.frontier.masked_dpo import EnvironmentMaskedDPOTrainer
from train.utils import attach_lora, load_base_model, load_lora_adapter


def prepare_model(*, model_id: str, lora_config, resume_checkpoint: Path | None):
    if resume_checkpoint is not None:
        model, tokenizer = load_lora_adapter(
            resume_checkpoint, model_id=model_id, use_4bit=config.USE_4BIT
        )
        print(f"Resuming from {resume_checkpoint}", flush=True)
    else:
        model, tokenizer = load_base_model(model_id=model_id, use_4bit=config.USE_4BIT)
        model = attach_lora(model, lora_config)
    if config.GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    model.print_trainable_parameters()
    return model, tokenizer


def build_dpo_trainer(*, model, tokenizer, dataset, args, adapter_dir: Path, tensorboard_dir: Path, use_tensorboard: bool, save_total_limit: int) -> DPOTrainer:
    train_kwargs = {
        "output_dir": str(adapter_dir),
        "per_device_train_batch_size": config.PER_DEVICE_BATCH,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
    }
    if args.max_steps is not None:
        train_kwargs["max_steps"] = args.max_steps
    dpo_args = DPOConfig(
        **train_kwargs,
        learning_rate=args.lr,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=config.WARMUP_RATIO,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        save_total_limit=save_total_limit,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        ddp_find_unused_parameters=False,
        fp16=config.TRAIN_FP16,
        bf16=config.TRAIN_BF16,
        report_to="tensorboard" if use_tensorboard else "none",
        logging_dir=str(tensorboard_dir) if use_tensorboard else None,
        beta=args.beta,
        max_length=args.max_length,
    )
    return EnvironmentMaskedDPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[MinimalCheckpointCallback()],
    )


def extract_train_metrics(trainer: DPOTrainer) -> dict:
    metrics = {"global_step": trainer.state.global_step, "epoch": trainer.state.epoch}
    if trainer.state.log_history:
        last = trainer.state.log_history[-1]
        for key in ("train_runtime", "train_loss", "train_samples_per_second", "train_steps_per_second"):
            if key in last:
                metrics[key] = last[key]
    return metrics
