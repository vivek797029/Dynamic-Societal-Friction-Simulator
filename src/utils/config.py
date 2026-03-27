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
    resume_from_checkpoint: bool = True
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine_with_restarts"
    num_cycles: int = 3
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    max_seq_length: int = 4096
    packing: bool = True
    gradient_checkpointing: bool = True
    save_strategy: str = "steps"
    save_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_total_limit: int = 10
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    logging_steps: int = 10
    logging_first_step: bool = True
    report_to: str = "wandb"
    optim: str = "paged_adamw_32bit"
    neftune_noise_alpha: float = 5.0
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    torch_compile: bool = False
    early_stopping: bool = True
    early_stopping_patience: int = 10


class DataConfig(BaseModel):
    """Data configuration including generation and augmentation settings."""
    train_file: str = "./data/processed/train.jsonl"
    eval_file: str = "./data/processed/eval.jsonl"
    prompt_template: str | None = "./data/prompts/friction_simulation.txt"
    max_samples: int | None = None
    generation: dict = Field(default_factory=dict)


class WandBConfig(BaseModel):
    """Weights & Biases logging configuration."""
    project: str = "dsfs-model-gcp"
    entity: str | None = None
    run_name: str | None = None


class GDriveConfig(BaseModel):
    """Google Drive checkpoint sync configuration."""
    enabled: bool = True
    sync_dir: str = "/content/drive/MyDrive/dsfs-checkpoints"
    sync_every_n_steps: int = 100


class ModelConfiguration(BaseModel):
    """Full model configuration schema with validation."""
    base_model: BaseModelConfig
    quantization: QuantizationConfig
    lora: LoraAdapterConfig
    training: TrainingConfig
    data: DataConfig = Field(default_factory=DataConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    gdrive: GDriveConfig = Field(default_factory=GDriveConfig)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load and return a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def validate_model_config(config_path: str) -> ModelConfiguration:
    """Load and validate model configuration."""
    raw = load_yaml_config(config_path)
    return ModelConfiguration(**raw)
