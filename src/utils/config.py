"""Configuration loading and validation utilities."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BaseModelConfig(BaseModel):
    name: str
    revision: str = "main"
    trust_remote_code: bool = False


class QuantizationConfig(BaseModel):
    enabled: bool = True
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


class LoraAdapterConfig(BaseModel):
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingConfig(BaseModel):
    output_dir: str = "./outputs/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    bf16: bool = True


class ModelConfiguration(BaseModel):
    """Full model configuration schema with validation."""
    base_model: BaseModelConfig
    quantization: QuantizationConfig
    lora: LoraAdapterConfig
    training: TrainingConfig


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load and return a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def validate_model_config(config_path: str) -> ModelConfiguration:
    """Load and validate model configuration."""
    raw = load_yaml_config(config_path)
    return ModelConfiguration(**raw)
