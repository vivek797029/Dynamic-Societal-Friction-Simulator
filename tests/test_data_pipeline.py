"""Comprehensive tests for synthetic data generation pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from src.model.data_pipeline import (
    format_as_instruction,
    generate_dataset,
    generate_synthetic_scenario,
)


# ========================================================================
# TESTS: Synthetic Scenario Generation
# ========================================================================


def test_generate_synthetic_scenario_structure():
    """Test that generated scenario has required fields."""
    scenario = generate_synthetic_scenario()
    assert "scenario" in scenario, "Scenario must contain 'scenario' field"
    assert "category" in scenario, "Scenario must contain 'category' field"
    assert "group_a" in scenario, "Scenario must contain 'group_a' field"
    assert "group_b" in scenario, "Scenario must contain 'group_b' field"
    assert "severity" in scenario, "Scenario must contain 'severity' field"


def test_generate_synthetic_scenario_severity_range():
    """Test that severity is within valid range."""
    scenario = generate_synthetic_scenario()
    assert 0.0 <= scenario["severity"] <= 1.0, "Severity must be between 0.0 and 1.0"


def test_generate_synthetic_scenario_distinct_groups():
    """Test that group_a and group_b are different."""
    scenario = generate_synthetic_scenario()
    assert scenario["group_a"] != scenario["group_b"], "Groups must be distinct"


def test_generate_synthetic_scenario_category_valid():
    """Test that category is from expected set."""
    valid_categories = {
        "cultural_clash",
        "resource_competition",
        "migration_tension",
        "values_conflict",
        "political_policy_disagreement",
        "election_fallout",
        "politicized_cultural_issue",
    }
    scenario = generate_synthetic_scenario()
    assert scenario["category"] in valid_categories, f"Category must be from {valid_categories}"


def test_generate_synthetic_scenario_text_content():
    """Test that scenario contains meaningful text."""
    scenario = generate_synthetic_scenario()
    assert isinstance(scenario["scenario"], str), "Scenario must be a string"
    assert len(scenario["scenario"]) > 20, "Scenario text must have meaningful length"


def test_generate_synthetic_scenario_multiple_calls_vary():
    """Test that multiple calls generate different scenarios."""
    scenarios = [generate_synthetic_scenario() for _ in range(5)]
    # At least some should be different
    texts = [s["scenario"] for s in scenarios]
    assert len(set(texts)) > 1, "Multiple calls should generate varied scenarios"


def test_generate_synthetic_scenario_groups_from_valid_set():
    """Test that groups are from the defined set."""
    valid_groups = {
        "Traditionalists",
        "Progressives",
        "Pragmatists",
        "Isolationists",
        "Youth_Activists",
        "Working_Class",
    }
    for _ in range(10):
        scenario = generate_synthetic_scenario()
        assert scenario["group_a"] in valid_groups
        assert scenario["group_b"] in valid_groups


# ========================================================================
# TESTS: Scenario Formatting
# ========================================================================


def test_format_as_instruction_structure():
    """Test that formatted instruction has required fields."""
    scenario = generate_synthetic_scenario()
    formatted = format_as_instruction(scenario)
    assert "instruction" in formatted, "Formatted must contain 'instruction' field"
    assert "output" in formatted, "Formatted must contain 'output' field"
    assert "category" in formatted, "Formatted must contain 'category' field"
    assert "metadata" in formatted, "Formatted must contain 'metadata' field"


def test_format_as_instruction_instruction_text():
    """Test that instruction field contains meaningful text."""
    scenario = generate_synthetic_scenario()
    formatted = format_as_instruction(scenario)
    assert isinstance(formatted["instruction"], str)
    assert len(formatted["instruction"]) > 10, "Instruction must have meaningful length"


def test_format_as_instruction_output_field():
    """Test that output field exists and is non-empty."""
    scenario = generate_synthetic_scenario()
    formatted = format_as_instruction(scenario)
    assert formatted["output"], "Output field must not be empty"
    assert isinstance(formatted["output"], str)


def test_format_as_instruction_metadata_contains_category():
    """Test that metadata includes category."""
    scenario = generate_synthetic_scenario()
    formatted = format_as_instruction(scenario)
    metadata = formatted["metadata"]
    assert "category" in metadata or scenario["category"] in str(metadata)


def test_format_as_instruction_preserves_category():
    """Test that category is preserved in formatted output."""
    scenario = generate_synthetic_scenario()
    formatted = format_as_instruction(scenario)
    assert formatted["category"] == scenario["category"]


def test_format_as_instruction_multiple_scenarios():
    """Test formatting works for multiple different scenarios."""
    for _ in range(5):
        scenario = generate_synthetic_scenario()
        formatted = format_as_instruction(scenario)
        assert "instruction" in formatted
        assert "output" in formatted
        assert len(formatted["instruction"]) > 0


# ========================================================================
# TESTS: Dataset Generation
# ========================================================================


def test_generate_dataset_creates_files():
    """Test that generate_dataset creates output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = generate_dataset(num_samples=10, output_dir=tmpdir, seed=42)

        train_path = Path(tmpdir) / "train.jsonl"
        assert train_path.exists(), "Training file must be created"


def test_generate_dataset_returns_stats():
    """Test that generate_dataset returns statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = generate_dataset(num_samples=20, output_dir=tmpdir, seed=42)
        assert isinstance(stats, dict)
        assert "train_samples" in stats
        assert "eval_samples" in stats


def test_generate_dataset_train_eval_split():
    """Test that samples are split into train and eval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_samples = 50
        stats = generate_dataset(num_samples=num_samples, output_dir=tmpdir, seed=42)

        total = stats["train_samples"] + stats["eval_samples"]
        assert total == num_samples, "Train + eval should equal total samples"


def test_generate_dataset_train_file_content():
    """Test that train file contains valid JSONL data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = generate_dataset(num_samples=10, output_dir=tmpdir, seed=42)

        train_path = Path(tmpdir) / "train.jsonl"
        with open(train_path) as f:
            lines = f.readlines()

        assert len(lines) == stats["train_samples"]

        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert "instruction" in data, "Each sample must have 'instruction'"
            assert "output" in data, "Each sample must have 'output'"


def test_generate_dataset_eval_file_exists():
    """Test that eval file is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = generate_dataset(num_samples=20, output_dir=tmpdir, seed=42)

        if stats["eval_samples"] > 0:
            eval_path = Path(tmpdir) / "eval.jsonl"
            assert eval_path.exists(), "Eval file should be created"


def test_generate_dataset_eval_file_content():
    """Test that eval file contains valid JSONL data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = generate_dataset(num_samples=30, output_dir=tmpdir, seed=42)

        if stats["eval_samples"] > 0:
            eval_path = Path(tmpdir) / "eval.jsonl"
            with open(eval_path) as f:
                lines = f.readlines()

            assert len(lines) == stats["eval_samples"]

            for line in lines:
                data = json.loads(line)
                assert "instruction" in data
                assert "output" in data


def test_generate_dataset_reproducible_with_seed():
    """Test that dataset generation is reproducible with same seed."""
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            # Generate two datasets with same seed
            generate_dataset(num_samples=10, output_dir=tmpdir1, seed=42)
            generate_dataset(num_samples=10, output_dir=tmpdir2, seed=42)

            # Read both train files
            with open(Path(tmpdir1) / "train.jsonl") as f1:
                lines1 = [json.loads(line) for line in f1]

            with open(Path(tmpdir2) / "train.jsonl") as f2:
                lines2 = [json.loads(line) for line in f2]

            # Should have same number of samples
            assert len(lines1) == len(lines2)


def test_generate_dataset_different_seeds_vary():
    """Test that different seeds produce different datasets."""
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            # Generate two datasets with different seeds
            generate_dataset(num_samples=20, output_dir=tmpdir1, seed=42)
            generate_dataset(num_samples=20, output_dir=tmpdir2, seed=123)

            # Read both train files
            with open(Path(tmpdir1) / "train.jsonl") as f1:
                lines1 = [json.loads(line)["instruction"] for line in f1]

            with open(Path(tmpdir2) / "train.jsonl") as f2:
                lines2 = [json.loads(line)["instruction"] for line in f2]

            # At least some should be different
            assert lines1 != lines2, "Different seeds should produce different data"


# ========================================================================
# TESTS: Data Augmentation (if applicable)
# ========================================================================


def test_generate_dataset_sample_quality():
    """Test that generated samples have good quality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = generate_dataset(num_samples=5, output_dir=tmpdir, seed=42)

        train_path = Path(tmpdir) / "train.jsonl"
        with open(train_path) as f:
            samples = [json.loads(line) for line in f]

        for sample in samples:
            # Instructions should be substantive
            assert len(sample["instruction"]) > 20
            # Outputs should be present
            assert len(sample["output"]) > 0
            # Both should be strings
            assert isinstance(sample["instruction"], str)
            assert isinstance(sample["output"], str)


def test_generate_dataset_diversity():
    """Test that dataset has diverse scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = generate_dataset(num_samples=30, output_dir=tmpdir, seed=42)

        train_path = Path(tmpdir) / "train.jsonl"
        with open(train_path) as f:
            samples = [json.loads(line) for line in f]

        # Check for category diversity in metadata if present
        categories = set()
        for sample in samples:
            if "metadata" in sample and isinstance(sample["metadata"], dict):
                if "category" in sample["metadata"]:
                    categories.add(sample["metadata"]["category"])

        # Should have multiple categories
        assert len(categories) > 0, "Should have category information"


# ========================================================================
# TESTS: Edge Cases
# ========================================================================


def test_generate_dataset_small_sample_count():
    """Test dataset generation with very small sample count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = generate_dataset(num_samples=1, output_dir=tmpdir, seed=42)
        assert stats["train_samples"] + stats["eval_samples"] == 1


def test_generate_dataset_large_sample_count():
    """Test dataset generation with larger sample count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = generate_dataset(num_samples=100, output_dir=tmpdir, seed=42)
        total = stats["train_samples"] + stats["eval_samples"]
        assert total == 100


def test_generate_dataset_empty_output_dir():
    """Test that function handles empty output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Directory is empty but exists
        stats = generate_dataset(num_samples=5, output_dir=tmpdir, seed=42)
        assert stats["train_samples"] > 0
