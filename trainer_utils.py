"""
Trainer Utilities for SFT Training Pipeline
Custom trainer class with additional features for SFT training
"""

import logging
import math
import os
import time
from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer,
    EvalPrediction
)
from transformers.trainer_utils import PredictionOutput
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import wandb


class SFTTrainer(Trainer):
    """Custom Trainer class for Supervised Fine-Tuning"""

    def __init__(
            self,
            model: PreTrainedModel,
            args: TrainingArguments,
            train_dataset=None,
            eval_dataset=None,
            data_collator=None,
            tokenizer: PreTrainedTokenizer = None,
            compute_metrics=None,
            **kwargs
    ):
        # Set up custom metrics if not provided
        if compute_metrics is None:
            compute_metrics = self._compute_metrics

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            **kwargs
        )

        self.logger = logging.getLogger(__name__)
        self.start_time = None

    def _compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute custom metrics for evaluation"""
        predictions, labels = eval_pred

        # Reshape if needed
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Convert to numpy if tensor
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        # Calculate perplexity from logits
        shift_predictions = predictions[..., :-1, :].reshape(-1, predictions.shape[-1])
        shift_labels = labels[..., 1:].reshape(-1)

        # Filter out ignored labels (-100)
        valid_mask = shift_labels != -100
        valid_predictions = shift_predictions[valid_mask]
        valid_labels = shift_labels[valid_mask]

        if len(valid_labels) == 0:
            return {"eval_loss": float('inf'), "perplexity": float('inf')}

        # Calculate cross-entropy loss
        log_probs = torch.nn.functional.log_softmax(torch.tensor(valid_predictions), dim=-1)
        nll_loss = torch.nn.functional.nll_loss(
            log_probs, torch.tensor(valid_labels), reduction='mean'
        )

        perplexity = torch.exp(nll_loss).item()

        # Calculate token-level accuracy
        predicted_tokens = np.argmax(valid_predictions, axis=-1)
        token_accuracy = accuracy_score(valid_labels, predicted_tokens)

        metrics = {
            "perplexity": perplexity,
            "token_accuracy": token_accuracy,
        }

        # Add sequence-level metrics if possible
        try:
            # This is a simplified sequence accuracy (exact match)
            seq_predictions = np.argmax(predictions, axis=-1)
            seq_accuracy = self._calculate_sequence_accuracy(seq_predictions, labels)
            metrics["sequence_accuracy"] = seq_accuracy
        except Exception as e:
            self.logger.warning(f"Could not calculate sequence accuracy: {e}")

        return metrics

    def _calculate_sequence_accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calculate sequence-level accuracy"""
        correct_sequences = 0
        total_sequences = predictions.shape[0]

        for i in range(total_sequences):
            pred_seq = predictions[i]
            label_seq = labels[i]

            # Find valid positions (not -100)
            valid_positions = label_seq != -100

            if not valid_positions.any():
                continue

            # Check if prediction matches labels at valid positions
            if np.array_equal(pred_seq[valid_positions], label_seq[valid_positions]):
                correct_sequences += 1

        return correct_sequences / total_sequences if total_sequences > 0 else 0.0

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """Custom training step with additional logging"""
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_apex:
            with amp.autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.use_cpu_amp:
            with self.scaler.scale():
                loss.backward()
        elif self.deepspeed:
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with additional metrics"""
        # Add timing information
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            logs["elapsed_time"] = elapsed_time

            if "train_loss" in logs and self.state.global_step > 0:
                # Calculate samples per second
                samples_per_second = self.state.global_step * self.args.train_batch_size / elapsed_time
                logs["samples_per_second"] = samples_per_second

        # Add learning rate
        if hasattr(self.lr_scheduler, 'get_last_lr'):
            logs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]

        # Add GPU memory usage if available
        if torch.cuda.is_available():
            logs["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
            logs["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)

        super().log(logs)

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, **kwargs):
        """Custom train method with timing"""
        self.start_time = time.time()
        self.logger.info("Starting training...")

        # Log initial model state
        if self.args.local_rank <= 0:  # Only log from main process
            self._log_model_info()

        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    def _log_model_info(self):
        """Log model information at the start of training"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100.0 * trainable_params / total_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }

        self.logger.info(f"Model info: {info}")

        # Log to wandb if available
        if self.args.report_to == "wandb" and wandb.run is not None:
            wandb.log(info, step=0)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        """Enhanced evaluation with additional metrics"""
        start_time = time.time()

        # Call parent evaluate method
        eval_results = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        # Add evaluation timing
        eval_time = time.time() - start_time
        eval_results[f"{metric_key_prefix}_time"] = eval_time

        # Log evaluation results
        self.logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{key}: {value:.4f}")

        return eval_results

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Enhanced model saving with metadata"""
        if output_dir is None:
            output_dir = self.args.output_dir

        # Save model using parent method
        super().save_model(output_dir, _internal_call)

        # Save additional metadata
        if self.args.local_rank <= 0:  # Only save from main process
            self._save_training_metadata(output_dir)

    def _save_training_metadata(self, output_dir: str):
        """Save training metadata and configuration"""
        import json
        from pathlib import Path

        metadata = {
            "training_steps": self.state.global_step,
            "training_epochs": self.state.epoch,
            "best_metric": getattr(self.state, 'best_metric', None),
            "best_model_checkpoint": getattr(self.state, 'best_model_checkpoint', None),
            "total_flos": self.state.total_flos,
        }

        # Add timing information
        if self.start_time is not None:
            metadata["total_training_time"] = time.time() - self.start_time

        # Save metadata
        metadata_path = Path(output_dir) / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Training metadata saved to {metadata_path}")

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix: str = "test") -> PredictionOutput:
        """Enhanced prediction with timing"""
        start_time = time.time()

        predictions = super().predict(
            test_dataset=test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        predict_time = time.time() - start_time
        self.logger.info(f"Prediction completed in {predict_time:.2f} seconds")

        return predictions


class TrainingCallback:
    """Custom callback for additional training monitoring"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        self.logger.info("Training started")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        self.logger.info("Training completed")

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        self.logger.info(f"Starting epoch {state.epoch}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        self.logger.info(f"Completed epoch {state.epoch}")

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        if state.global_step % (args.logging_steps * 10) == 0:  # Log every 10th logging step
            self.logger.debug(f"Completed step {state.global_step}")


def calculate_optimal_batch_size(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        num_gpus: int = 1,
        target_memory_utilization: float = 0.85
) -> int:
    """
    Calculate optimal batch size based on model size and available GPU memory

    Args:
        model: The model to train
        tokenizer: The tokenizer
        max_seq_length: Maximum sequence length
        num_gpus: Number of GPUs
        target_memory_utilization: Target GPU memory utilization (0.0-1.0)

    Returns:
        Optimal batch size per device
    """
    if not torch.cuda.is_available():
        return 1

    # Get GPU memory info
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    available_memory = total_memory * target_memory_utilization

    # Estimate model memory usage
    model_params = sum(p.numel() for p in model.parameters())

    # Rough estimates (in bytes)
    # Model parameters (4 bytes per parameter for float32)
    model_memory = model_params * 4

    # Gradients (same size as parameters)
    gradient_memory = model_params * 4

    # Optimizer states (Adam uses ~8 bytes per parameter)
    optimizer_memory = model_params * 8

    # Base memory usage
    base_memory = model_memory + gradient_memory + optimizer_memory

    # Memory per sample (rough estimate)
    # Input tokens + attention + intermediate activations
    tokens_per_sample = max_seq_length
    memory_per_token = 4 * model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4 * 768
    memory_per_sample = tokens_per_sample * memory_per_token * 4  # Factor for activations

    # Calculate batch size
    remaining_memory = available_memory - base_memory
    if remaining_memory <= 0:
        return 1

    batch_size = max(1, int(remaining_memory // memory_per_sample))

    # Cap at reasonable maximum
    batch_size = min(batch_size, 64)

    return batch_size


def setup_wandb_logging(config, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """Setup Weights & Biases logging"""
    if not hasattr(config, 'report_to') or config.report_to != 'wandb':
        return

    try:
        import wandb

        # Initialize wandb run
        wandb.init(
            project=getattr(config, 'wandb_project', 'sft-training'),
            name=getattr(config, 'run_name', None),
            config={
                'model_name': config.model_name_or_path,
                'learning_rate': config.learning_rate,
                'batch_size': config.per_device_train_batch_size,
                'num_epochs': config.num_train_epochs,
                'max_seq_length': config.max_seq_length,
                'gradient_accumulation_steps': config.gradient_accumulation_steps,
            }
        )

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        wandb.log({
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
            'model/trainable_percentage': 100.0 * trainable_params / total_params,
            'tokenizer/vocab_size': tokenizer.vocab_size,
        }, step=0)

    except ImportError:
        logging.getLogger(__name__).warning("wandb not installed, skipping W&B logging")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to setup wandb logging: {e}")