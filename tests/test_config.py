"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    BaseModelConfig,
    DataConfig,
    LoraAdapterConfig,
    ModelConfiguration,
    QuantizationConfig,
    TrainingConfig,
    load_yaml_config,
    validate_model_config,
)


# ========================================================================
# TESTS: YAML Config Loading
# ========================================================================


class TestYamlConfigLoading:
    """Tests for loading YAML configuration files."""

    def test_load_yaml_config_basic(self):
        """Test loading basic YAML config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "key1": "value1",
                "key2": 42,
                "nested": {"inner_key": "inner_value"},
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            loaded = load_yaml_config(config_path)
            assert loaded["key1"] == "value1"
            assert loaded["key2"] == 42
            assert loaded["nested"]["inner_key"] == "inner_value"

    def test_load_yaml_config_string_path(self):
        """Test loading with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {"test": "data"}

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            # Load with string path
            loaded = load_yaml_config(str(config_path))
            assert loaded["test"] == "data"

    def test_load_yaml_config_path_object(self):
        """Test loading with Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {"test": "data"}

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            # Load with Path object
            loaded = load_yaml_config(config_path)
            assert loaded["test"] == "data"

    def test_load_yaml_config_missing_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/path/config.yaml")

    def test_load_yaml_config_complex_structure(self):
        """Test loading complex nested YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "level1": {
                    "level2": {
                        "level3": {
                            "value": "deep_value",
                        }
                    },
                    "list": [1, 2, 3],
                },
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            loaded = load_yaml_config(config_path)
            assert loaded["level1"]["level2"]["level3"]["value"] == "deep_value"
            assert loaded["level1"]["list"] == [1, 2, 3]


# ========================================================================
# TESTS: Config Classes
# ========================================================================


class TestBaseModelConfig:
    """Tests for BaseModelConfig."""

    def test_base_model_required_fields(self):
        """Test BaseModelConfig with required fields."""
        config = BaseModelConfig(name="bert-base-uncased")
        assert config.name == "bert-base-uncased"
        assert config.revision == "main"
        assert config.trust_remote_code is False

    def test_base_model_custom_revision(self):
        """Test BaseModelConfig with custom revision."""
        config = BaseModelConfig(name="gpt2", revision="v1.0")
        assert config.revision == "v1.0"

    def test_base_model_trust_code(self):
        """Test BaseModelConfig with trust_remote_code."""
        config = BaseModelConfig(name="custom", trust_remote_code=True)
        assert config.trust_remote_code is True

    def test_base_model_missing_name_raises(self):
        """Test that missing name raises validation error."""
        with pytest.raises(ValueError):
            BaseModelConfig()


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_quantization_defaults(self):
        """Test quantization with default values."""
        config = QuantizationConfig()
        assert config.enabled is True
        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == "bfloat16"

    def test_quantization_custom_values(self):
        """Test quantization with custom values."""
        config = QuantizationConfig(
            enabled=False,
            load_in_4bit=False,
            bnb_4bit_compute_dtype="float32",
        )
        assert config.enabled is False
        assert config.load_in_4bit is False
        assert config.bnb_4bit_compute_dtype == "float32"


class TestLoraAdapterConfig:
    """Tests for LoraAdapterConfig."""

    def test_lora_defaults(self):
        """Test LoRA config with defaults."""
        config = LoraAdapterConfig()
        assert config.r == 64
        assert config.lora_alpha == 128
        assert config.lora_dropout == 0.05
        assert "q_proj" in config.target_modules

    def test_lora_custom_rank(self):
        """Test LoRA config with custom rank."""
        config = LoraAdapterConfig(r=128, lora_alpha=256)
        assert config.r == 128
        assert config.lora_alpha == 256


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_training_defaults(self):
        """Test training config with defaults."""
        config = TrainingConfig()
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 4
        assert config.bf16 is True

    def test_training_custom_epochs(self):
        """Test training with custom epoch count."""
        config = TrainingConfig(num_train_epochs=5)
        assert config.num_train_epochs == 5

    def test_training_learning_rate(self):
        """Test training learning rate setting."""
        config = TrainingConfig(learning_rate=1e-4)
        assert config.learning_rate == 1e-4


class TestDataConfig:
    """Tests for DataConfig."""

    def test_data_defaults(self):
        """Test data config with defaults."""
        config = DataConfig()
        assert "train" in config.train_file
        assert "eval" in config.eval_file

    def test_data_custom_paths(self):
        """Test data config with custom paths."""
        config = DataConfig(
            train_file="./custom_train.jsonl",
            eval_file="./custom_eval.jsonl",
        )
        assert config.train_file == "./custom_train.jsonl"
        assert config.eval_file == "./custom_eval.jsonl"

    def test_data_max_samples(self):
        """Test data config with max_samples."""
        config = DataConfig(max_samples=1000)
        assert config.max_samples == 1000


# ========================================================================
# TESTS: ModelConfiguration Validation
# ========================================================================


class TestModelConfigurationValidation:
    """Tests for full model configuration validation."""

    def test_valid_complete_config(self):
        """Test validation of complete valid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "base_model": {
                    "name": "meta-llama/Llama-2-7b",
                    "revision": "main",
                },
                "quantization": {
                    "enabled": True,
                },
                "lora": {
                    "r": 64,
                },
                "training": {
                    "num_train_epochs": 3,
                },
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            config = validate_model_config(str(config_path))
            assert config.base_model.name == "meta-llama/Llama-2-7b"
            assert config.quantization.enabled is True

    def test_config_missing_base_model(self):
        """Test validation fails without base_model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "quantization": {"enabled": True},
                "lora": {"r": 64},
                "training": {"num_train_epochs": 3},
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            with pytest.raises(ValueError):
                validate_model_config(str(config_path))

    def test_config_missing_quantization(self):
        """Test validation fails without quantization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "base_model": {"name": "test"},
                "lora": {"r": 64},
                "training": {"num_train_epochs": 3},
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            with pytest.raises(ValueError):
                validate_model_config(str(config_path))

    def test_config_missing_training(self):
        """Test validation fails without training config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "base_model": {"name": "test"},
                "quantization": {"enabled": True},
                "lora": {"r": 64},
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            with pytest.raises(ValueError):
                validate_model_config(str(config_path))

    def test_config_with_optional_fields(self):
        """Test config with optional wandb and gdrive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "base_model": {"name": "test"},
                "quantization": {"enabled": True},
                "lora": {"r": 64},
                "training": {"num_train_epochs": 3},
                "wandb": {"project": "my-project"},
                "gdrive": {"enabled": False},
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            config = validate_model_config(str(config_path))
            assert config.wandb.project == "my-project"
            assert config.gdrive.enabled is False


# ========================================================================
# TESTS: Config Edge Cases
# ========================================================================


class TestConfigEdgeCases:
    """Tests for edge cases in config handling."""

    def test_config_with_extra_fields_ignored(self):
        """Test that extra fields in config are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "base_model": {"name": "test", "extra_field": "ignored"},
                "quantization": {"enabled": True},
                "lora": {"r": 64},
                "training": {"num_train_epochs": 3},
                "unknown_section": {"data": "ignored"},
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            # Pydantic should ignore extra fields by default
            config = validate_model_config(str(config_path))
            assert config.base_model.name == "test"

    def test_config_with_invalid_types(self):
        """Test config with invalid field types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "base_model": {"name": "test"},
                "quantization": {"enabled": True},
                "lora": {"r": "invalid_string"},  # Should be int
                "training": {"num_train_epochs": 3},
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            with pytest.raises(ValueError):
                validate_model_config(str(config_path))

    def test_config_with_boolean_strings(self):
        """Test config with YAML boolean strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            # YAML treats yes/no as booleans
            yaml_content = """
base_model:
  name: test
  trust_remote_code: yes
quantization:
  enabled: no
lora:
  r: 64
training:
  num_train_epochs: 3
"""
            with open(config_path, "w") as f:
                f.write(yaml_content)

            config = validate_model_config(str(config_path))
            assert config.base_model.trust_remote_code is True
            assert config.quantization.enabled is False

    def test_data_config_with_none_max_samples(self):
        """Test data config with None max_samples."""
        config = DataConfig(max_samples=None)
        assert config.max_samples is None

    def test_training_config_learning_rate_formats(self):
        """Test training config with different learning rate formats."""
        config1 = TrainingConfig(learning_rate=1e-4)
        assert config1.learning_rate == 0.0001

        config2 = TrainingConfig(learning_rate=0.0002)
        assert config2.learning_rate == 0.0002
