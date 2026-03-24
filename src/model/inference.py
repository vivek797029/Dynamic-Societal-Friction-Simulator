"""
Inference module for the fine-tuned Society Friction LLM.

Loads the QLoRA adapter on top of the base model and provides
a clean generation API used by simulation agents.
"""

import logging
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class FrictionLLM:
    """Wrapper around the fine-tuned friction model for agent inference."""

    def __init__(
        self,
        config_path: str = "configs/model_config.yaml",
        adapter_path: str | None = None,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self._load_model(adapter_path)

    def _load_model(self, adapter_path: str | None):
        """Load base model + LoRA adapter."""
        model_name = self.cfg["base_model"]["name"]
        qcfg = self.cfg["quantization"]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qcfg["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load adapter if available
        if adapter_path and Path(adapter_path).exists():
            logger.info(f"Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            logger.warning("No adapter loaded — using base model only.")

        model.eval()
        return model, tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate a response from the friction LLM."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def analyze_friction(self, scenario: str, group_a: str, group_b: str) -> dict:
        """Analyze a friction scenario between two groups."""
        prompt = (
            f"<|scenario|>\n{scenario}\n"
            f"<|group_a|>\n{group_a}\n"
            f"<|group_b|>\n{group_b}\n"
            f"<|analysis|>\n"
        )
        raw = self.generate(prompt, max_new_tokens=1024, temperature=0.4)
        return {"raw_analysis": raw, "scenario": scenario, "groups": [group_a, group_b]}

    def predict_escalation(self, history: list[str], current_event: str) -> str:
        """Predict how a friction event might escalate given history."""
        history_block = "\n".join(f"- {h}" for h in history[-10:])
        prompt = (
            f"<|history|>\n{history_block}\n"
            f"<|current_event|>\n{current_event}\n"
            f"<|prediction|>\n"
        )
        return self.generate(prompt, max_new_tokens=512, temperature=0.5)
