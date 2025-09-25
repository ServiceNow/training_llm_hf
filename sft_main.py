#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Training Pipeline for Large Language Models
Author: Applied Research Scientist
Description: Modular, distributed training pipeline with HuggingFace integration
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, get_scheduler
)
from datasets import Dataset, load_dataset, load_from_disk
import wandb
from dataclasses import dataclass, field
import yaml

# Custom modules
from data_handler import DataHandler
from model_manager import ModelManager
from trainer_utils import SFTTrainer
from config_manager import ConfigManager
from distributed_utils import setup_distributed, cleanup_distributed


@dataclass
class SFTConfig:
    """Configuration class for SFT training"""
    # Model configurations
    model_name_or_path: str = "microsoft/DialoGPT-medium"
    tokenizer_name: Optional[str] = None
    model_revision: str = "main"
    use_auth_token: bool = False

    # Data configurations
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    dataset_path: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    max_seq_length: int = 512
    preprocessing_num_workers: int = 4

    # Training configurations
    output_dir: str = "./sft_output"
    overwrite_output_dir: bool = True
    do_train: bool = True
    do_eval: bool = True
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 500
    lr_scheduler_type: str = "linear"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Distributed training configurations
    local_rank: int = -1
    world_size: int = 1
    num_gpus: int = 1
    ddp_backend: str = "nccl"

    # Optimization configurations
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Logging and monitoring
    logging_dir: str = "./logs"
    report_to: str = "wandb"
    run_name: Optional[str] = None

    # Advanced configurations
    resume_from_checkpoint: Optional[str] = None
    seed: int = 42


class SFTTrainingPipeline:
    """Main SFT Training Pipeline"""

    def __init__(self, config: SFTConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.device = None
        self.is_distributed = False
        self.local_rank = -1

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        logger = logging.getLogger(__name__)
        return logger

    def setup_distributed_training(self):
        """Setup distributed training environment"""
        if self.config.num_gpus > 1:
            self.is_distributed = True
            self.local_rank, world_size = setup_distributed(self.config.ddp_backend)
            self.config.world_size = world_size
            self.config.local_rank = self.local_rank
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            self.logger.info(f"Distributed training setup: rank {self.local_rank}, world_size {world_size}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Single GPU/CPU training on {self.device}")

    def calculate_batch_sizes(self) -> tuple:
        """Calculate optimal batch sizes based on available resources"""
        # Get GPU memory info
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.logger.info(f"GPU memory available: {gpu_memory_gb:.2f} GB")
        else:
            gpu_memory_gb = 8  # Default assumption for CPU

        # Adjust batch size based on model size and GPU memory
        base_batch_size = self.config.per_device_train_batch_size

        # Simple heuristic for batch size adjustment
        if gpu_memory_gb < 12:  # For smaller GPUs
            recommended_batch_size = max(1, base_batch_size // 2)
        elif gpu_memory_gb > 24:  # For larger GPUs
            recommended_batch_size = base_batch_size * 2
        else:
            recommended_batch_size = base_batch_size

        # Calculate effective batch size with gradient accumulation
        effective_batch_size = (recommended_batch_size *
                                self.config.gradient_accumulation_steps *
                                self.config.world_size)

        self.logger.info(f"Per-device batch size: {recommended_batch_size}")
        self.logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        self.logger.info(f"Effective batch size: {effective_batch_size}")

        return recommended_batch_size, effective_batch_size

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        model_manager = ModelManager(self.config)
        model, tokenizer = model_manager.load_model_and_tokenizer()

        if self.is_distributed:
            model = model.to(self.device)
            model = DDP(model, device_ids=[self.local_rank])
        else:
            model = model.to(self.device)

        return model, tokenizer

    def prepare_datasets(self, tokenizer):
        """Prepare training and validation datasets"""
        data_handler = DataHandler(self.config, tokenizer)
        train_dataset, eval_dataset = data_handler.prepare_datasets()

        self.logger.info(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            self.logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def setup_training_arguments(self, per_device_batch_size: int):
        """Setup HuggingFace training arguments"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=self.config.overwrite_output_dir,
            do_train=self.config.do_train,
            do_eval=self.config.do_eval,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
            max_grad_norm=self.config.max_grad_norm,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            logging_dir=self.config.logging_dir,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
            local_rank=self.config.local_rank,
            seed=self.config.seed,
            ddp_backend=self.config.ddp_backend,
            resume_from_checkpoint=self.config.resume_from_checkpoint,
        )

        return training_args

    def train(self):
        """Main training function"""
        self.logger.info("Starting SFT training pipeline...")

        # Setup distributed training
        self.setup_distributed_training()

        # Calculate optimal batch sizes
        per_device_batch_size, effective_batch_size = self.calculate_batch_sizes()

        # Setup model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer()

        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets(tokenizer)

        # Setup training arguments
        training_args = self.setup_training_arguments(per_device_batch_size)

        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # For causal LM
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Start training
        if self.config.do_train:
            self.logger.info("Starting training...")
            train_result = trainer.train(
                resume_from_checkpoint=self.config.resume_from_checkpoint
            )

            # Save model and tokenizer
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)

            # Log training results
            self.logger.info(f"Training completed. Results: {train_result}")

        # Evaluation
        if self.config.do_eval:
            self.logger.info("Starting evaluation...")
            eval_results = trainer.evaluate()
            self.logger.info(f"Evaluation results: {eval_results}")

        # Cleanup
        if self.is_distributed:
            cleanup_distributed()

        self.logger.info("SFT training pipeline completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SFT Training Pipeline")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to configuration file (JSON or YAML)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )

    args = parser.parse_args()

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config_file, SFTConfig)

    # Override local_rank if provided
    if args.local_rank != -1:
        config.local_rank = args.local_rank

    # Initialize and run training pipeline
    pipeline = SFTTrainingPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()