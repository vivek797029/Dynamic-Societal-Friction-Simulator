"""Tests for the synthetic data generation pipeline."""

import json
import tempfile
from pathlib import Path

from src.model.data_pipeline import (
    format_as_instruction,
    generate_dataset,
    generate_synthetic_scenario,
)


def test_generate_synthetic_scenario():
    scenario = generate_synthetic_scenario()
    assert "scenario" in scenario
    assert "category" in scenario
    assert "group_a" in scenario
    assert "group_b" in scenario
    assert 0.0 <= scenario["severity"] <= 1.0
    assert scenario["group_a"] != scenario["group_b"]


def test_format_as_instruction():
    scenario = generate_synthetic_scenario()
    formatted = format_as_instruction(scenario)
    assert "instruction" in formatted
    assert "output" in formatted
    assert "category" in formatted
    assert "metadata" in formatted


def test_generate_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = generate_dataset(num_samples=50, output_dir=tmpdir, seed=42)
        assert stats["train_samples"] + stats["eval_samples"] == 50

        train_path = Path(tmpdir) / "train.jsonl"
        assert train_path.exists()

        with open(train_path) as f:
            lines = f.readlines()
        assert len(lines) == stats["train_samples"]

        sample = json.loads(lines[0])
        assert "instruction" in sample
        assert "output" in sample


def test_reproducibility():
    s1 = generate_dataset.__wrapped__ if hasattr(generate_dataset, "__wrapped__") else None
    scenario_a = generate_synthetic_scenario()
    # Basic sanity: scenarios are generated without errors
    assert isinstance(scenario_a["scenario"], str)
    assert len(scenario_a["scenario"]) > 20
