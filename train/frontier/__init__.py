"""Model and DPO trainer construction for frontier-paraphrased pairs."""

from .engine import build_dpo_trainer, extract_train_metrics, prepare_model
from .masked_dpo import EnvironmentMaskedDPOTrainer, environment_spans

__all__ = [
    "EnvironmentMaskedDPOTrainer",
    "build_dpo_trainer",
    "environment_spans",
    "extract_train_metrics",
    "prepare_model",
]
