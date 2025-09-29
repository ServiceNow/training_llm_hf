"""
Configuration Manager for SFT Training Pipeline
Handles loading and validation of configuration files
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Type, TypeVar
from dataclasses import fields, is_dataclass
import logging

T = TypeVar('T')


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str, config_class: Type[T]) -> T:
        """
        Load configuration from JSON or YAML file

        Args:
            config_path: Path to configuration file
            config_class: Dataclass to instantiate with loaded config

        Returns:
            Instance of config_class with loaded parameters
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load based on file extension
        if config_path.suffix.lower() == '.json':
            config_dict = self._load_json(config_path)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            config_dict = self._load_yaml(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        # Validate and create config instance
        config_instance = self._create_config_instance(config_dict, config_class)

        self.logger.info(f"Configuration loaded from {config_path}")
        return config_instance

    def _load_json(self, config_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")

    def _load_yaml(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}")

    def _create_config_instance(self, config_dict: Dict[str, Any], config_class: Type[T]) -> T:
        """Create config instance from dictionary with validation"""
        if not is_dataclass(config_class):
            raise ValueError("config_class must be a dataclass")

        # Get valid field names
        valid_fields = {field.name for field in fields(config_class)}

        # Filter out invalid keys and warn about them
        filtered_config = {}
        for key, value in config_dict.items():
            if key in valid_fields:
                filtered_config[key] = value
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")

        # Create instance
        try:
            return config_class(**filtered_config)
        except TypeError as e:
            raise ValueError(f"Error creating configuration instance: {e}")

    def save_config(self, config_instance: Any, output_path: str, format: str = 'yaml'):
        """
        Save configuration instance to file

        Args:
            config_instance: Configuration instance to save
            output_path: Path to save configuration
            format: Output format ('json' or 'yaml')
        """
        output_path = Path(output_path)

        if not is_dataclass(config_instance):
            raise ValueError("config_instance must be a dataclass instance")

        # Convert to dictionary
        config_dict = {}
        for field in fields(config_instance):
            config_dict[field.name] = getattr(config_instance, field.name)

        # Save based on format
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif format.lower() in ['yaml', 'yml']:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Configuration saved to {output_path}")


def create_sample_config(output_path: str = "config.yaml"):
    """Create a sample configuration file"""
    sample_config = {
        # Model configurations
        "model_name_or_path": "microsoft/DialoGPT-medium",
        "tokenizer_name": None,
        "model_revision": "main",
        "use_auth_token": False,

        # Data configurations
        "dataset_name": None,
        "dataset_config_name": None,
        "dataset_path": "./data",
        "train_file": "train.jsonl",
        "validation_file": "validation.jsonl",
        "max_seq_length": 512,
        "preprocessing_num_workers": 4,

        # Training configurations
        "output_dir": "./sft_output",
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "num_train_epochs": 3,
        "max_steps": -1,
        "warmup_steps": 500,
        "lr_scheduler_type": "linear",
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 3,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,

        # Distributed training configurations
        "num_gpus": 1,
        "ddp_backend": "nccl",

        # Optimization configurations
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,

        # Logging and monitoring
        "logging_dir": "./logs",
        "report_to": "wandb",
        "run_name": "sft_training_run",

        # Advanced configurations
        "resume_from_checkpoint": None,
        "seed": 42
    }

    with open(output_path, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)

    print(f"Sample configuration created at: {output_path}")


if __name__ == "__main__":
    create_sample_config()