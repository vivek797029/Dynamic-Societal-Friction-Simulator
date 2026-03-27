"""
QLoRA Fine-Tuning Pipeline — A100 Optimized, 4-Hour Budget.

Trains Mistral-7B-Instruct-v0.3 on social/political friction data using
4-bit quantized LoRA with full checkpoint resume support.
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
try:
    from trl import SFTTrainer, SFTConfig as _SFTConfig
    _HAS_SFT_CONFIG = True
except ImportError:
    from trl import SFTTrainer
    _HAS_SFT_CONFIG = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# Model / Tokenizer
# ─────────────────────────────────────────────────────────────

def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    l = cfg["lora"]
    return LoraConfig(
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        lora_dropout=l["lora_dropout"],
        target_modules=l["target_modules"],
        bias=l["bias"],
        task_type=l["task_type"],
    )


def load_model_and_tokenizer(cfg: dict, bnb_config: BitsAndBytesConfig):
    name = cfg["base_model"]["name"]
    logger.info(f"Loading model: {name}")

    # Use Flash Attention 2 if available (A100 supports it)
    attn_impl = "eager"
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg["base_model"].get("trust_remote_code", False),
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────

def load_data(cfg: dict):
    d = cfg["data"]
    dataset = load_dataset(
        "json",
        data_files={"train": d["train_file"], "eval": d["eval_file"]},
    )
    if d.get("max_samples"):
        n = d["max_samples"]
        dataset["train"] = dataset["train"].select(range(min(n, len(dataset["train"]))))
        dataset["eval"]  = dataset["eval"].select(range(min(n // 5, len(dataset["eval"]))))
    logger.info(f"Train: {len(dataset['train'])}  Eval: {len(dataset['eval'])}")
    return dataset


# ─────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str) -> str | None:
    pattern = str(Path(output_dir) / "checkpoint-*")
    dirs = sorted(
        glob.glob(pattern),
        key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1))
        if re.search(r"checkpoint-(\d+)", x) else 0,
    )
    if dirs:
        logger.info(f"Resuming from: {dirs[-1]}")
        return dirs[-1]
    return None


def sync_to_gdrive(cfg: dict, output_dir: str):
    gdrive = cfg.get("gdrive", {})
    if not gdrive.get("enabled", False):
        return
    drive_dir = Path(gdrive.get("sync_dir", "/content/drive/MyDrive/dsfs-checkpoints"))
    if not Path("/content/drive").exists():
        return
    try:
        drive_dir.mkdir(parents=True, exist_ok=True)
        ckpt = find_latest_checkpoint(output_dir)
        if ckpt:
            dst = drive_dir / Path(ckpt).name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(ckpt, dst)
            logger.info(f"Synced to Drive: {dst}")
    except Exception as e:
        logger.warning(f"Drive sync failed (non-fatal): {e}")


def restore_from_gdrive(cfg: dict, output_dir: str) -> str | None:
    gdrive = cfg.get("gdrive", {})
    if not gdrive.get("enabled", False):
        return None
    drive_dir = Path(gdrive.get("sync_dir", "/content/drive/MyDrive/dsfs-checkpoints"))
    if not drive_dir.exists() or find_latest_checkpoint(output_dir):
        return None
    dirs = sorted(glob.glob(str(drive_dir / "checkpoint-*")),
                  key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1))
                  if re.search(r"checkpoint-(\d+)", x) else 0)
    if not dirs:
        return None
    dst = Path(output_dir) / Path(dirs[-1]).name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copytree(dirs[-1], dst)
    logger.info(f"Restored from Drive: {dst}")
    return str(dst)


# ─────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────

class GDriveSyncCallback(TrainerCallback):
    def __init__(self, cfg: dict, sync_every_n_steps: int = 100):
        self.cfg = cfg
        self.sync_every_n_steps = sync_every_n_steps
        self.output_dir = cfg["training"]["output_dir"]

    def on_save(self, args, state, control, **kwargs):
        if state.global_step % self.sync_every_n_steps == 0:
            sync_to_gdrive(self.cfg, self.output_dir)

    def on_train_end(self, args, state, control, **kwargs):
        sync_to_gdrive(self.cfg, self.output_dir)


# ─────────────────────────────────────────────────────────────
# Training Args  (uses TrainingArguments — avoids SFTConfig
#                 version-compatibility issues with max_seq_length)
# ─────────────────────────────────────────────────────────────

def build_training_args(tcfg: dict):
    """Build training args. Uses SFTConfig if available (puts max_seq_length/packing there)."""

    # Base args that exist in all transformers versions
    args = dict(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg.get("weight_decay", 0.01),
        warmup_ratio=tcfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        fp16=tcfg.get("fp16", False),
        bf16=tcfg.get("bf16", True),
        logging_steps=tcfg.get("logging_steps", 10),
        logging_first_step=tcfg.get("logging_first_step", True),
        save_strategy=tcfg.get("save_strategy", "steps"),
        save_steps=tcfg.get("save_steps", 100),
        eval_strategy=tcfg.get("eval_strategy", "steps"),
        eval_steps=tcfg.get("eval_steps", 100),
        save_total_limit=tcfg.get("save_total_limit", 3),
        load_best_model_at_end=tcfg.get("load_best_model_at_end", True),
        metric_for_best_model=tcfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=tcfg.get("greater_is_better", False),
        report_to=tcfg.get("report_to", "none"),
        optim=tcfg.get("optim", "paged_adamw_32bit"),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
    )

    # NEFTune — only in transformers >= 4.35
    neftune = tcfg.get("neftune_noise_alpha")
    if neftune:
        try:
            import inspect
            from transformers import TrainingArguments as _TA
            if "neftune_noise_alpha" in inspect.signature(_TA.__init__).parameters:
                args["neftune_noise_alpha"] = neftune
        except Exception:
            pass

    # If SFTConfig is available, add max_seq_length + packing there
    if _HAS_SFT_CONFIG:
        import inspect as _i
        _sft_cfg_params = _i.signature(_SFTConfig.__init__).parameters
        if "max_seq_length" in _sft_cfg_params:
            args["max_seq_length"] = tcfg.get("max_seq_length", 2048)
        if "packing" in _sft_cfg_params:
            args["packing"] = tcfg.get("packing", True)
        return _SFTConfig(**args)

    return TrainingArguments(**args)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def train(config_path: str = "configs/model_config.yaml", resume: bool | None = None):
    cfg  = load_config(config_path)
    tcfg = cfg["training"]
    output_dir = tcfg["output_dir"]

    # ── Checkpoint resume ────────────────────────────────────
    should_resume = tcfg.get("resume_from_checkpoint", True) if resume is None else resume
    resume_ckpt   = None
    if should_resume:
        resume_ckpt = find_latest_checkpoint(output_dir) or restore_from_gdrive(cfg, output_dir)
        logger.info(f"Resume checkpoint: {resume_ckpt or 'None (fresh start)'}")

    # ── GPU info ─────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu} ({mem:.1f} GB)")
    else:
        logger.warning("No GPU — training will be very slow")

    # ── Build model ──────────────────────────────────────────
    bnb_cfg   = build_bnb_config(cfg)
    lora_cfg  = build_lora_config(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg, bnb_cfg)
    model = get_peft_model(model, lora_cfg)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Data ─────────────────────────────────────────────────
    dataset = load_data(cfg)

    # ── Training args ────────────────────────────────────────
    training_args = build_training_args(tcfg)
    max_seq_length = tcfg.get("max_seq_length", 2048)

    # ── Callbacks ────────────────────────────────────────────
    callbacks = []
    gdrive_cfg = cfg.get("gdrive", {})
    if gdrive_cfg.get("enabled", False) and Path("/content/drive").exists():
        interval = gdrive_cfg.get("sync_every_n_steps", 100)
        callbacks.append(GDriveSyncCallback(cfg, sync_every_n_steps=interval))
        logger.info(f"Drive sync every {interval} steps")

    # ── W&B ──────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg and os.environ.get("WANDB_DISABLED", "false").lower() != "true":
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg.get("project", "dsfs"))

    # ── Formatting function ───────────────────────────────────
    # Our data has 'instruction' + 'output' fields, not 'text'.
    # This converts each sample into a single string for SFTTrainer.
    def formatting_func(example):
        instruction = example.get("instruction", "")
        inp         = example.get("input", "")
        output      = example.get("output", "")
        if inp:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    # ── Trainer ──────────────────────────────────────────────
    # Auto-detect what SFTTrainer accepts across all TRL versions
    import inspect as _inspect
    _sft_sig = _inspect.signature(SFTTrainer.__init__).parameters

    _trainer_kwargs: dict = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        callbacks=callbacks,
        formatting_func=formatting_func,
    )

    # tokenizer vs processing_class (TRL >= 0.9 renamed it)
    if "processing_class" in _sft_sig:
        _trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in _sft_sig:
        _trainer_kwargs["tokenizer"] = tokenizer

    # max_seq_length and packing only if SFTTrainer still accepts them
    if "max_seq_length" in _sft_sig:
        _trainer_kwargs["max_seq_length"] = max_seq_length
    if "packing" in _sft_sig:
        _trainer_kwargs["packing"] = tcfg.get("packing", True)

    trainer = SFTTrainer(**_trainer_kwargs)

    # ── Train ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info(f"  Epochs:     {tcfg['num_train_epochs']}")
    logger.info(f"  Eff. batch: {tcfg['per_device_train_batch_size'] * tcfg['gradient_accumulation_steps']}")
    logger.info(f"  Seq len:    {max_seq_length}")
    logger.info(f"  LoRA r:     {cfg['lora']['r']}")
    logger.info("=" * 60)

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    elapsed = time.time() - t0
    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)
    logger.info(f"Training done in {int(h)}h {int(m)}m {int(s)}s")

    # ── Save ─────────────────────────────────────────────────
    final = Path(output_dir) / "final_adapter"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    logger.info(f"Saved: {final}")

    # Final Drive sync
    sync_to_gdrive(cfg, output_dir)
    if gdrive_cfg.get("enabled", False) and Path("/content/drive").exists():
        drive_final = Path(gdrive_cfg["sync_dir"]) / "final_adapter"
        try:
            if drive_final.exists():
                shutil.rmtree(drive_final)
            shutil.copytree(final, drive_final)
            logger.info(f"Final adapter → Drive: {drive_final}")
        except Exception as e:
            logger.warning(f"Drive final sync failed: {e}")

    return trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
