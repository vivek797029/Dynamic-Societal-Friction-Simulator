"""
QLoRA Fine-Tuning Pipeline for the Society Friction LLM.

GCP CLOUD MODE — Full checkpoint resume support.
Trains Mistral-7B on social/cultural/political friction data using
4-bit quantized LoRA. Designed for A100/V100/T4 on Google Cloud / Colab.

Key features:
    - Auto-detect and resume from latest checkpoint
    - Google Drive checkpoint sync for Colab persistence
    - Early stopping with configurable patience
    - Cosine-with-restarts LR scheduler
    - NEFTune noise injection
    - Comprehensive logging via W&B
"""

import glob
import logging
import os
import re
import shutil
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================
# Quantization & LoRA
# ============================================================

def build_quantization_config(cfg: dict) -> BitsAndBytesConfig:
    """Build BitsAndBytes config for 4-bit QLoRA."""
    qcfg = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    """Build LoRA adapter configuration."""
    lcfg = cfg["lora"]
    return LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        bias=lcfg["bias"],
        task_type=lcfg["task_type"],
    )


# ============================================================
# Model Loading
# ============================================================

def load_base_model(cfg: dict, bnb_config: BitsAndBytesConfig):
    """Load and prepare the base model with quantization."""
    model_name = cfg["base_model"]["name"]
    logger.info(f"Loading base model: {model_name}")

    # Detect flash attention support
    attn_impl = "eager"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Flash Attention 2 detected — using for faster training")
    except ImportError:
        logger.info("Flash Attention not available — using eager attention")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg["base_model"].get("trust_remote_code", False),
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ============================================================
# Data Loading
# ============================================================

def load_training_data(cfg: dict):
    """Load train/eval datasets from JSONL files."""
    data_cfg = cfg["data"]
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_file"],
            "eval": data_cfg["eval_file"],
        },
    )
    if data_cfg.get("max_samples"):
        dataset["train"] = dataset["train"].select(
            range(min(data_cfg["max_samples"], len(dataset["train"])))
        )
        dataset["eval"] = dataset["eval"].select(
            range(min(data_cfg["max_samples"] // 5, len(dataset["eval"])))
        )

    logger.info(
        f"Training samples: {len(dataset['train'])}, "
        f"Eval samples: {len(dataset['eval'])}"
    )
    return dataset


# ============================================================
# Checkpoint Management
# ============================================================

def find_latest_checkpoint(output_dir: str) -> str | None:
    """
    Find the latest checkpoint in the output directory.

    Checkpoints are named like 'checkpoint-500', 'checkpoint-1000', etc.
    Returns the path to the latest one (highest step number), or None.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.info(f"Output directory {output_dir} does not exist — starting fresh")
        return None

    checkpoint_dirs = sorted(
        glob.glob(str(output_path / "checkpoint-*")),
        key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1))
        if re.search(r"checkpoint-(\d+)", x)
        else 0,
    )

    if not checkpoint_dirs:
        logger.info("No checkpoints found — starting fresh")
        return None

    latest = checkpoint_dirs[-1]
    step_num = re.search(r"checkpoint-(\d+)", latest)
    logger.info(
        f"Found {len(checkpoint_dirs)} checkpoint(s). "
        f"Resuming from: {latest} (step {step_num.group(1) if step_num else '?'})"
    )
    return latest


def sync_checkpoints_to_gdrive(cfg: dict, output_dir: str):
    """
    Sync checkpoint files to Google Drive for persistence across Colab sessions.

    Only runs if gdrive.enabled is true in config.
    """
    gdrive_cfg = cfg.get("gdrive", {})
    if not gdrive_cfg.get("enabled", False):
        return

    sync_dir = gdrive_cfg.get("sync_dir", "/content/drive/MyDrive/dsfs-checkpoints")
    sync_path = Path(sync_dir)

    # Check if Google Drive is mounted
    if not Path("/content/drive").exists():
        logger.warning("Google Drive not mounted — skipping checkpoint sync")
        return

    try:
        sync_path.mkdir(parents=True, exist_ok=True)

        # Copy latest checkpoint
        latest_ckpt = find_latest_checkpoint(output_dir)
        if latest_ckpt:
            dest = sync_path / Path(latest_ckpt).name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(latest_ckpt, dest)
            logger.info(f"Checkpoint synced to Google Drive: {dest}")

        # Also copy trainer_state.json for resume metadata
        trainer_state = Path(output_dir) / "trainer_state.json"
        if trainer_state.exists():
            shutil.copy2(trainer_state, sync_path / "trainer_state.json")

    except Exception as e:
        logger.warning(f"Google Drive sync failed (non-fatal): {e}")


def restore_from_gdrive(cfg: dict, output_dir: str) -> str | None:
    """
    If local checkpoints are missing but Google Drive has them,
    restore from Drive. This handles Colab session restarts.

    Returns path to restored checkpoint, or None.
    """
    gdrive_cfg = cfg.get("gdrive", {})
    if not gdrive_cfg.get("enabled", False):
        return None

    sync_dir = gdrive_cfg.get("sync_dir", "/content/drive/MyDrive/dsfs-checkpoints")
    sync_path = Path(sync_dir)

    if not sync_path.exists():
        return None

    # Check if local checkpoints exist
    local_latest = find_latest_checkpoint(output_dir)
    if local_latest:
        return None  # Local checkpoints exist, no need to restore

    # Find checkpoints on Drive
    drive_checkpoints = sorted(
        glob.glob(str(sync_path / "checkpoint-*")),
        key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1))
        if re.search(r"checkpoint-(\d+)", x)
        else 0,
    )

    if not drive_checkpoints:
        return None

    latest_drive = drive_checkpoints[-1]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Restore checkpoint from Drive
    dest = output_path / Path(latest_drive).name
    logger.info(f"Restoring checkpoint from Google Drive: {latest_drive} → {dest}")
    shutil.copytree(latest_drive, dest)

    # Restore trainer_state.json if available
    drive_state = sync_path / "trainer_state.json"
    if drive_state.exists():
        shutil.copy2(drive_state, output_path / "trainer_state.json")

    return str(dest)


# ============================================================
# Custom Callback for Google Drive Sync
# ============================================================

class GDriveSyncCallback(TrainerCallback):
    """
    HuggingFace Trainer callback that syncs checkpoints to Google Drive
    at configurable intervals.
    """

    def __init__(self, cfg: dict, sync_every_n_steps: int = 100):
        self.cfg = cfg
        self.sync_every_n_steps = sync_every_n_steps
        self.output_dir = cfg["training"]["output_dir"]

    def on_save(self, args, state, control, **kwargs):
        """Called after a checkpoint is saved."""
        if state.global_step % self.sync_every_n_steps == 0:
            sync_checkpoints_to_gdrive(self.cfg, self.output_dir)

    def on_train_end(self, args, state, control, **kwargs):
        """Sync final checkpoint when training ends."""
        sync_checkpoints_to_gdrive(self.cfg, self.output_dir)


# ============================================================
# Training Arguments Builder
# ============================================================

def build_training_args(tcfg: dict) -> SFTConfig:
    """
    Build SFTConfig from config, handling all GCP-optimized settings.
    SFTConfig extends TrainingArguments with max_seq_length and packing.
    """
    args_dict = {
        "output_dir": tcfg["output_dir"],
        "num_train_epochs": tcfg["num_train_epochs"],
        "per_device_train_batch_size": tcfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": tcfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": tcfg["gradient_accumulation_steps"],
        "learning_rate": tcfg["learning_rate"],
        "weight_decay": tcfg["weight_decay"],
        "warmup_ratio": tcfg["warmup_ratio"],
        "lr_scheduler_type": tcfg["lr_scheduler_type"],
        "max_grad_norm": tcfg["max_grad_norm"],
        "fp16": tcfg["fp16"],
        "bf16": tcfg["bf16"],
        "logging_steps": tcfg["logging_steps"],
        "logging_first_step": tcfg.get("logging_first_step", True),
        "save_strategy": tcfg["save_strategy"],
        "save_steps": tcfg["save_steps"],
        "eval_strategy": tcfg["eval_strategy"],
        "eval_steps": tcfg["eval_steps"],
        "save_total_limit": tcfg["save_total_limit"],
        "load_best_model_at_end": tcfg["load_best_model_at_end"],
        "metric_for_best_model": tcfg["metric_for_best_model"],
        "greater_is_better": tcfg.get("greater_is_better", False),
        "report_to": tcfg["report_to"],
        "optim": tcfg.get("optim", "paged_adamw_32bit"),
        "gradient_checkpointing": tcfg.get("gradient_checkpointing", True),
        "dataloader_num_workers": tcfg.get("dataloader_num_workers", 4),
        "dataloader_pin_memory": tcfg.get("dataloader_pin_memory", True),
        "torch_compile": tcfg.get("torch_compile", False),
        # SFT-specific args (moved from SFTTrainer to SFTConfig in newer TRL)
        "max_seq_length": tcfg.get("max_seq_length", 2048),
        "packing": tcfg.get("packing", True),
    }

    # NEFTune noise alpha (if supported)
    if "neftune_noise_alpha" in tcfg:
        args_dict["neftune_noise_alpha"] = tcfg["neftune_noise_alpha"]

    return SFTConfig(**args_dict)


# ============================================================
# Main Training Function
# ============================================================

def train(config_path: str = "configs/model_config.yaml", resume: bool | None = None):
    """
    Main training entrypoint with full checkpoint resume support.

    Args:
        config_path: Path to model_config.yaml
        resume: If True, force resume from checkpoint. If False, start fresh.
                If None (default), auto-detect: resume if checkpoints exist.
    """
    cfg = load_config(config_path)
    tcfg = cfg["training"]
    output_dir = tcfg["output_dir"]

    # ---- Determine resume behavior ----
    should_resume = tcfg.get("resume_from_checkpoint", True) if resume is None else resume

    resume_checkpoint = None
    if should_resume:
        # First try local checkpoints
        resume_checkpoint = find_latest_checkpoint(output_dir)

        # If no local checkpoints, try restoring from Google Drive
        if resume_checkpoint is None:
            resume_checkpoint = restore_from_gdrive(cfg, output_dir)

        if resume_checkpoint:
            logger.info(f"Will resume training from: {resume_checkpoint}")
        else:
            logger.info("No checkpoints found anywhere — starting from scratch")

    # ---- GPU Info ----
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("No GPU detected! Training will be extremely slow on CPU.")

    # ---- Build components ----
    bnb_config = build_quantization_config(cfg)
    lora_config = build_lora_config(cfg)
    model, tokenizer = load_base_model(cfg, bnb_config)
    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    # ---- Data ----
    dataset = load_training_data(cfg)

    # Load prompt template
    template_path = cfg["data"].get("prompt_template")
    prompt_template = None
    if template_path and Path(template_path).exists():
        prompt_template = Path(template_path).read_text()

    # ---- Training arguments ----
    training_args = build_training_args(tcfg)

    # ---- Callbacks ----
    callbacks = []

    # Early stopping
    if tcfg.get("early_stopping", False):
        patience = tcfg.get("early_stopping_patience", 5)
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        logger.info(f"Early stopping enabled with patience={patience}")

    # Google Drive sync callback
    gdrive_cfg = cfg.get("gdrive", {})
    if gdrive_cfg.get("enabled", False) and Path("/content/drive").exists():
        sync_interval = gdrive_cfg.get("sync_every_n_steps", 100)
        callbacks.append(GDriveSyncCallback(cfg, sync_every_n_steps=sync_interval))
        logger.info(f"Google Drive sync enabled every {sync_interval} steps")

    # ---- W&B Setup ----
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg:
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg.get("project", "dsfs-model-gcp"))
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = wandb_cfg["entity"]
        if wandb_cfg.get("run_name"):
            os.environ["WANDB_RUN_NAME"] = wandb_cfg["run_name"]

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # ---- Train (with resume support) ----
    train_start = time.time()
    logger.info("=" * 60)
    if resume_checkpoint:
        logger.info(f"RESUMING training from checkpoint: {resume_checkpoint}")
        logger.info("Optimizer state, LR scheduler, and step count will be restored.")
    else:
        logger.info("STARTING fresh training run")
    logger.info(f"  Epochs: {tcfg['num_train_epochs']}")
    logger.info(f"  Batch size: {tcfg['per_device_train_batch_size']} x {tcfg['gradient_accumulation_steps']} = {tcfg['per_device_train_batch_size'] * tcfg['gradient_accumulation_steps']} effective")
    logger.info(f"  Learning rate: {tcfg['learning_rate']}")
    logger.info(f"  Max seq length: {tcfg['max_seq_length']}")
    logger.info(f"  Scheduler: {tcfg['lr_scheduler_type']}")
    logger.info("=" * 60)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    train_elapsed = time.time() - train_start
    hours, remainder = divmod(train_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # ---- Save final adapter ----
    final_path = Path(output_dir) / "final_adapter"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Final adapter saved to: {final_path}")

    # ---- Final sync to Google Drive ----
    sync_checkpoints_to_gdrive(cfg, output_dir)

    # Also save final adapter to Drive
    if gdrive_cfg.get("enabled", False) and Path("/content/drive").exists():
        drive_final = Path(gdrive_cfg["sync_dir"]) / "final_adapter"
        try:
            if drive_final.exists():
                shutil.rmtree(drive_final)
            shutil.copytree(final_path, drive_final)
            logger.info(f"Final adapter synced to Google Drive: {drive_final}")
        except Exception as e:
            logger.warning(f"Failed to sync final adapter to Drive: {e}")

    return trainer


# ============================================================
# Standalone entrypoint
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
