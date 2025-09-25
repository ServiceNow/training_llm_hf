"""
Model Manager for SFT Training Pipeline
Handles model and tokenizer loading, configuration, and optimization
"""

import logging
from typing import Tuple, Optional, Dict, Any
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import warnings


class ModelManager:
    """Manages model and tokenizer loading and configuration"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load and configure model and tokenizer

        Returns:
            Tuple of (model, tokenizer)
        """
        # Load tokenizer
        tokenizer = self._load_tokenizer()

        # Load model
        model = self._load_model()

        # Configure model for training
        model = self._configure_model_for_training(model, tokenizer)

        # Apply optimizations
        model = self._apply_optimizations(model)

        return model, tokenizer

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and configure tokenizer"""
        tokenizer_name = self.config.tokenizer_name or self.config.model_name_or_path

        self.logger.info(f"Loading tokenizer: {tokenizer_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                revision=self.config.model_revision,
                use_auth_token=self.config.use_auth_token,
                trust_remote_code=True,
            )
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise

        # Configure tokenizer
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Ensure left padding for generation (if needed)
        tokenizer.padding_side = "right"  # Use right padding for training

        self.logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        self.logger.info(
            f"Special tokens - PAD: {tokenizer.pad_token}, EOS: {tokenizer.eos_token}, BOS: {tokenizer.bos_token}")

        return tokenizer

    def _load_model(self) -> PreTrainedModel:
        """Load model with appropriate configuration"""
        self.logger.info(f"Loading model: {self.config.model_name_or_path}")

        # Configure model loading arguments
        model_kwargs = {
            "revision": self.config.model_revision,
            "use_auth_token": self.config.use_auth_token,
            "trust_remote_code": True,
        }

        # Configure precision
        if self.config.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        elif self.config.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        # Configure device map for multi-GPU
        if self.config.num_gpus > 1 and not torch.distributed.is_initialized():
            model_kwargs["device_map"] = "auto"

        # Load with quantization if specified
        quantization_config = self._get_quantization_config()
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                **model_kwargs
            )
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

        self.logger.info(f"Model loaded. Parameters: {self._count_parameters(model):,}")

        return model

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration if needed"""
        # Add quantization support if config specifies it
        quantization_config = getattr(self.config, 'quantization_config', None)

        if quantization_config:
            if quantization_config.get('load_in_4bit', False):
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=quantization_config.get('bnb_4bit_quant_type', 'nf4'),
                    bnb_4bit_compute_dtype=getattr(torch, quantization_config.get('bnb_4bit_compute_dtype', 'float16')),
                    bnb_4bit_use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', True),
                )
            elif quantization_config.get('load_in_8bit', False):
                return BitsAndBytesConfig(load_in_8bit=True)

        return None

    def _configure_model_for_training(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        """Configure model for training"""

        # Resize token embeddings if tokenizer was modified
        if len(tokenizer) != model.config.vocab_size:
            self.logger.info(f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

        # Configure for training
        model.train()

        # Enable gradient checkpointing if specified
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")

        return model

    def _apply_optimizations(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply model optimizations"""

        # Apply LoRA if configured
        lora_config = getattr(self.config, 'lora_config', None)
        if lora_config:
            model = self._apply_lora(model, lora_config)

        # Prepare for quantized training if needed
        if hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit:
            model = prepare_model_for_kbit_training(model)
            self.logger.info("Model prepared for 8-bit training")
        elif hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
            model = prepare_model_for_kbit_training(model)
            self.logger.info("Model prepared for 4-bit training")

        return model

    def _apply_lora(self, model: PreTrainedModel, lora_config: Dict[str, Any]) -> PreTrainedModel:
        """Apply LoRA (Low-Rank Adaptation) to the model"""
        self.logger.info("Applying LoRA configuration")

        peft_config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            bias=lora_config.get('bias', "none"),
            task_type=lora_config.get('task_type', "CAUSAL_LM"),
        )

        model = get_peft_model(model, peft_config)

        # Print trainable parameters
        trainable_params, all_params = self._count_peft_parameters(model)
        self.logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")

        return model

    def _count_parameters(self, model: PreTrainedModel) -> int:
        """Count total model parameters"""
        return sum(p.numel() for p in model.parameters())

    def _count_peft_parameters(self, model) -> Tuple[int, int]:
        """Count trainable and total parameters for PEFT models"""
        trainable_params = 0
        all_param = 0

        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return trainable_params, all_param

    def get_model_info(self, model: PreTrainedModel) -> Dict[str, Any]:
        """Get detailed model information"""
        info = {
            "model_type": model.config.model_type if hasattr(model.config, 'model_type') else 'unknown',
            "total_parameters": self._count_parameters(model),
            "vocab_size": model.config.vocab_size if hasattr(model.config, 'vocab_size') else 'unknown',
            "hidden_size": model.config.hidden_size if hasattr(model.config, 'hidden_size') else 'unknown',
            "num_layers": getattr(model.config, 'num_hidden_layers', getattr(model.config, 'n_layer', 'unknown')),
            "num_attention_heads": getattr(model.config, 'num_attention_heads',
                                           getattr(model.config, 'n_head', 'unknown')),
        }

        # Add PEFT info if applicable
        if hasattr(model, 'peft_config'):
            trainable_params, all_params = self._count_peft_parameters(model)
            info.update({
                "is_peft_model": True,
                "trainable_parameters": trainable_params,
                "trainable_percentage": 100 * trainable_params / all_params,
            })
        else:
            info["is_peft_model"] = False

        # Add quantization info
        if hasattr(model, 'is_loaded_in_8bit'):
            info["is_8bit"] = model.is_loaded_in_8bit
        if hasattr(model, 'is_loaded_in_4bit'):
            info["is_4bit"] = model.is_loaded_in_4bit

        return info


def print_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """Print detailed model and tokenizer information"""
    manager = ModelManager(None)
    model_info = manager.get_model_info(model)

    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    for key, value in model_info.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        elif isinstance(value, int) and value > 1000:
            print(f"{key.replace('_', ' ').title()}: {value:,}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")

    print("\nTOKENIZER INFORMATION")
    print("-" * 30)
    print(f"Vocab Size: {tokenizer.vocab_size:,}")
    print(f"Model Max Length: {tokenizer.model_max_length}")
    print(f"Pad Token: {tokenizer.pad_token}")
    print(f"EOS Token: {tokenizer.eos_token}")
    print(f"BOS Token: {tokenizer.bos_token}")
    print(f"UNK Token: {tokenizer.unk_token}")
    print("=" * 60 + "\n")