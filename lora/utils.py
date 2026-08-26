"""LoRA model-loading and paper adapter-merge utilities."""

from __future__ import annotations

import lora.gpu_env  # noqa: F401

import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config


def _check_bitsandbytes_cuda() -> None:
    import bitsandbytes.functional as bnb_fn
    import bitsandbytes.cextension as bnb_ext

    lib = bnb_ext.lib
    if lib is None or not getattr(lib, "compiled_with_cuda", False):
        raise RuntimeError(
            "bitsandbytes loaded CPU library (4-bit quantize unavailable). "
            "Load CUDA 12.6 and its math libraries before training "
            f"(BNB_CUDA_VERSION={os.environ.get('BNB_CUDA_VERSION', '')!r})"
        )
    probe = torch.randn(8, 8, device="cuda", dtype=torch.float16)
    try:
        bnb_fn.quantize_4bit(probe)
    except Exception as exc:
        raise RuntimeError(
            "bitsandbytes 4-bit probe failed on GPU. "
            "Run lora/train.sh from a CUDA GPU allocation."
        ) from exc


def build_bnb_config() -> BitsAndBytesConfig:
    compute_dtype = getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
    )


def build_lora_config(
    r: int | None = None,
    alpha: int | None = None,
    target_modules: list[str] | str | None = None,
) -> LoraConfig:
    tm = target_modules if target_modules is not None else config.LORA_TARGET_MODULES
    return LoraConfig(
        r=r or config.LORA_R,
        lora_alpha=alpha or config.LORA_ALPHA,
        target_modules=tm,
        lora_dropout=config.LORA_DROPOUT,
        task_type="CAULM",
    )


def load_base_model(
    model_id: str | None = None,
    use_4bit: bool = True,
) -> tuple[Any, Any]:
    model_id = model_id or config.MODEL_ID
    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit load requires a visible GPU (torch.cuda.is_available() is False). "
                "On the cluster run: module load cuda/12.6 && nvidia-smi"
            )
        _check_bitsandbytes_cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"device_map": "auto"}
    if use_4bit:
        kwargs["quantization_config"] = build_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    return model, tokenizer


def attach_lora(model, lora_config: LoraConfig | None = None):
    lora_config = lora_config or build_lora_config()
    return get_peft_model(model, lora_config)


def load_lora_adapter(
    adapter_path: str | Path,
    model_id: str | None = None,
    use_4bit: bool = True,
):
    model, tokenizer = load_base_model(model_id=model_id, use_4bit=use_4bit)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    return model, tokenizer


def merge_lora_adapters(
    aux_path: str | Path,
    all_path: str | Path,
    out_path: str | Path,
    alpha: float = 0.5,
    model_id: str | None = None,
) -> Path:
    """W_merge = alpha * W_AUX + (1-alpha) * W_ALL (paper Eq. 4)."""
    from safetensors.torch import load_file, save_file

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    aux_files = list(Path(aux_path).glob("*.safetensors"))
    all_files = list(Path(all_path).glob("*.safetensors"))
    if not aux_files or not all_files:
        raise FileNotFoundError("Adapter safetensors not found in aux/all paths")

    aux_sd = load_file(str(aux_files[0]))
    all_sd = load_file(str(all_files[0]))
    merged = {}
    for key in aux_sd:
        if key in all_sd:
            merged[key] = alpha * aux_sd[key] + (1.0 - alpha) * all_sd[key]
        else:
            merged[key] = aux_sd[key]

    save_file(merged, str(out_path / aux_files[0].name))
    return out_path
