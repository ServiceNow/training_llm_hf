"""
Data Handler for SFT Training Pipeline
Handles dataset loading, preprocessing, and tokenization
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from transformers import PreTrainedTokenizer
import pandas as pd
import random

class DataHandler:
    """Handles data loading and preprocessing for SFT training"""

    def __init__(self, config, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)

    # def prepare_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
    #     """
    #     Prepare training and validation datasets

    #     Returns:
    #         Tuple of (train_dataset, eval_dataset)
    #     """
    #     # Load raw datasets
    #     raw_datasets = self._load_datasets()

    #     # Preprocess datasets
    #     processed_datasets = self._preprocess_datasets(raw_datasets)

    #     # Extract train and validation splits
    #     train_dataset = processed_datasets.get('train')
    #     eval_dataset = processed_datasets.get('validation') or processed_datasets.get('test')

    #     if train_dataset is None:
    #         raise ValueError("No training dataset found")

    #     return train_dataset, eval_dataset

    def _load_datasets(self) -> DatasetDict:
        """Load datasets from various sources"""

        # Load from HuggingFace Hub
        if self.config.dataset_name:
            self.logger.info(f"Loading dataset from HuggingFace Hub: {self.config.dataset_name}")
            try:
                datasets = load_dataset(
                    self.config.dataset_name,
                    self.config.dataset_config_name,
                    use_auth_token=self.config.use_auth_token
                )
                return datasets
            except Exception as e:
                self.logger.error(f"Error loading dataset from Hub: {e}")
                raise

        # Load from local directory (saved HuggingFace dataset)
        elif self.config.dataset_path and Path(self.config.dataset_path).is_dir():
            self.logger.info(f"Loading dataset from local directory: {self.config.dataset_path}")
            try:
                datasets = load_from_disk(self.config.dataset_path)
                return datasets
            except Exception as e:
                self.logger.warning(f"Failed to load as HF dataset, trying as files: {e}")
                return self._load_from_files()

        # Load from individual files
        else:
            return self._load_from_files()

    def _load_from_files(self) -> DatasetDict:
        """Load datasets from individual files"""
        datasets = DatasetDict()

        # Load training file
        if self.config.train_file:
            train_path = Path(self.config.dataset_path or ".") / self.config.train_file
            if train_path.exists():
                self.logger.info(f"Loading training data from: {train_path}")
                train_data = self._load_single_file(train_path)
                datasets['train'] = Dataset.from_list(train_data)
            else:
                raise FileNotFoundError(f"Training file not found: {train_path}")

        # Load validation file
        if self.config.validation_file:
            val_path = Path(self.config.dataset_path or ".") / self.config.validation_file
            if val_path.exists():
                self.logger.info(f"Loading validation data from: {val_path}")
                val_data = self._load_single_file(val_path)
                datasets['validation'] = Dataset.from_list(val_data)
            else:
                self.logger.warning(f"Validation file not found: {val_path}")

        return datasets

    def _load_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from a single file"""
        file_extension = file_path.suffix.lower()

        try:
            if file_extension == '.jsonl':
                return self._load_jsonl(file_path)
            elif file_extension == '.json':
                return self._load_json(file_path)
            elif file_extension == '.csv':
                return self._load_csv(file_path)
            elif file_extension in ['.txt']:
                return self._load_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            raise

    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
        return data

    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ensure data is a list
        if isinstance(data, dict):
            # If it's a dict, try to extract the main data
            for key in ['data', 'examples', 'instances']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If no standard key found, wrap in list
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("JSON file must contain a list or dict with data")

    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV file"""
        df = pd.read_csv(file_path)
        return df.to_dict('records')

    def _load_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load plain text file (each line as a separate example)"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append({"text": line})
        return data

    def _preprocess_datasets(self, raw_datasets: DatasetDict) -> DatasetDict:
        """Preprocess and tokenize datasets"""
        processed_datasets = DatasetDict()

        for split, dataset in raw_datasets.items():
            self.logger.info(f"Preprocessing {split} dataset with {len(dataset)} examples")

            # Apply preprocessing function
            processed_dataset = dataset.map(
                self._preprocess_function,
                batched=True,
                num_proc=self.config.preprocessing_num_workers,
                remove_columns=dataset.column_names,
                desc=f"Preprocessing {split} dataset",
            )

            # Filter out examples that are too long
            processed_dataset = processed_dataset.filter(
                lambda example: len(example["input_ids"]) <= self.config.max_seq_length
            )

            processed_datasets[split] = processed_dataset
            self.logger.info(f"Processed {split} dataset: {len(processed_dataset)} examples")

        return processed_datasets

    def _preprocess_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        """
        Preprocess function for tokenization

        Expected input formats:
        1. {"text": [...]} - Simple text completion
        2. {"input": [...], "output": [...]} - Input-output pairs
        3. {"prompt": [...], "completion": [...]} - Prompt-completion pairs
        4. {"instruction": [...], "input": [...], "output": [...]} - Instruction format
        5. {"messages": [...]} - Chat format (list of message dicts)
        """
        texts = []

        for i in range(len(examples.get(list(examples.keys())[0], []))):
            example = {key: examples[key][i] for key in examples.keys()}
            text = self._format_example(example)
            texts.append(text)

        # Tokenize
        # Tokenize with truncation and padding
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",          # pad each sequence to max_seq_length
            max_length=self.config.max_seq_length,
            return_overflowing_tokens=False,
        )

        # For causal LM, labels are the same as input_ids
        # tokenized["labels"] = tokenized["input_ids"].copy()
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format a single example into text suitable for causal LM training"""

        # Handle different input formats
        if "messages" in example and example["messages"] is not None:
            # Chat format - convert messages to conversation
            text = self._format_chat_messages(example["messages"])

        elif "instruction" in example:
            # Instruction format (Alpaca-style)
            instruction = example["instruction"]
            input_text = example.get("input", "")
            output_text = example.get("output", "")

            if input_text:
                text = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{input_text}\n\n"
                    f"### Response:\n{output_text}"
                )
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"

        elif "prompt" in example and "completion" in example:
            # Prompt-completion format
            text = f"{example['prompt']}{example['completion']}"

        elif "input" in example and "output" in example:
            # Input-output format
            text = f"Input: {example['input']}\nOutput: {example['output']}"

        elif "text" in example:
            # Simple text format
            text = example["text"]

        else:
            # Fallback: concatenate all string values
            text_parts = []
            for key, value in example.items():
                if isinstance(value, str) and value.strip():
                    text_parts.append(f"{key}: {value}")
            text = "\n".join(text_parts)

        # Ensure text ends with EOS token
        if not text.endswith(self.tokenizer.eos_token):
            text += self.tokenizer.eos_token

        return text

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a conversation string"""
        formatted_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
            else:
                formatted_parts.append(f"{role.title()}: {content}")

        return "\n".join(formatted_parts)

    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """Get information about the dataset"""
        if len(dataset) == 0:
            return {"size": 0, "columns": [], "sample": None}

        sample = dataset[0]
        return {
            "size": len(dataset),
            "columns": dataset.column_names,
            "sample": sample,
            "avg_length": sum(len(example["input_ids"]) for example in dataset) / len(dataset),
            "max_length": max(len(example["input_ids"]) for example in dataset),
            "min_length": min(len(example["input_ids"]) for example in dataset),
        }
        
    def prepare_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Prepare training and validation datasets"""
        if hasattr(self.config, "datasets") and self.config.datasets:
            raw_train, raw_eval = self._load_multiple_datasets()
        else:
            raw_datasets = self._load_datasets()
            raw_train = raw_datasets.get("train")
            raw_eval = raw_datasets.get("validation") or raw_datasets.get("test")

        if raw_train is None:
            raise ValueError("No training dataset found")

        # Preprocess
        processed_train = self._preprocess_datasets(DatasetDict({"train": raw_train}))["train"]
        processed_eval = None
        if raw_eval:
            processed_eval = self._preprocess_datasets(DatasetDict({"validation": raw_eval}))["validation"]

        return processed_train, processed_eval

    def _load_multiple_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Load and mix multiple datasets based on config"""
        train_datasets, eval_datasets = [], []
        ratios = []

        for ds_cfg in self.config.datasets:
            if "name" in ds_cfg:  # Load from HF Hub
                ds = load_dataset(ds_cfg["name"], split=ds_cfg.get("split", "train"))
            elif "path" in ds_cfg:  # Local path
                if Path(ds_cfg["path"]).is_dir():
                    ds = load_from_disk(ds_cfg["path"])
                else:
                    # fallback to files
                    temp_handler = DataHandler(ds_cfg, self.tokenizer)
                    ds = temp_handler._load_from_files()
                ds = ds.get("train", None)
            else:
                raise ValueError(f"Invalid dataset config: {ds_cfg}")

            if ds is None:
                continue

            train_datasets.append(ds)
            ratios.append(ds_cfg.get("ratio", 1.0))

            # Load eval split if available
            if "validation_file" in ds_cfg or ds_cfg.get("split") == "validation":
                eval_datasets.append(ds)

        # Mix training datasets
        mixed_train = self._mix_datasets(train_datasets, ratios)

        mixed_train = mixed_train.select(range(max_train_samples)) if (max_train_samples := getattr(self.config, "max_train_samples", None)) else mixed_train
            

        mixed_eval = concatenate_datasets(eval_datasets) if eval_datasets else None
        return mixed_train, mixed_eval

    def _mix_datasets(self, datasets: List[Dataset], ratios: List[float]) -> Dataset:
        """Mix multiple datasets according to ratios"""
        # Normalize ratios
        total = sum(ratios)
        ratios = [r / total for r in ratios]

        lengths = [int(len(ds) * r) for ds, r in zip(datasets, ratios)]
        sampled = []

        for idx, (ds, target_len) in enumerate(zip(datasets, lengths)):
            # Tag dataset source
            ds = ds.add_column("dataset_id", [idx] * len(ds))

            if target_len < len(ds):
                idxs = random.sample(range(len(ds)), target_len)
                sampled.append(ds.select(idxs))
            else:
                sampled.append(ds)

        return concatenate_datasets(sampled)


# class DataCollatorForSFT:
#     """Custom data collator for SFT training with proper padding and attention masks"""

#     def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_multiple_of: Optional[int] = None):
#         self.tokenizer = tokenizer
#         self.pad_to_multiple_of = pad_to_multiple_of

#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         # Get the maximum length in the batch
#         max_length = max(len(feature["input_ids"]) for feature in features)

#         # Pad to multiple if specified
#         if self.pad_to_multiple_of is not None:
#             max_length = ((max_length + self.pad_to_multiple_of - 1)
#                           // self.pad_to_multiple_of * self.pad_to_multiple_of)

#         batch = {}
#         batch_size = len(features)

#         # Initialize tensors
#         batch["input_ids"] = torch.full((batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long)
#         batch["attention_mask"] = torch.zeros(batch_size, max_length, dtype=torch.long)
#         batch["labels"] = torch.full((batch_size, max_length), -100, dtype=torch.long)

#         # Fill tensors
#         for i, feature in enumerate(features):
#             input_ids = feature["input_ids"]
#             seq_length = len(input_ids)

#             batch["input_ids"][i, :seq_length] = torch.tensor(input_ids, dtype=torch.long)
#             batch["attention_mask"][i, :seq_length] = 1
#             batch["labels"][i, :seq_length] = torch.tensor(feature["labels"], dtype=torch.long)

#         return batch

class DataCollatorForSFT:
    """Custom data collator for SFT training with dataset tracking"""

    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Received empty features list in DataCollatorForSFT")

        # Determine max sequence length in batch
        max_length = max(len(f["input_ids"]) for f in features)

        if self.pad_to_multiple_of is not None:
            # Round up to nearest multiple
            max_length = ((max_length + self.pad_to_multiple_of - 1)
                          // self.pad_to_multiple_of * self.pad_to_multiple_of)

        batch_size = len(features)

        # Initialize tensors
        input_ids = torch.full((batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        dataset_ids = torch.tensor([f.get("dataset_id", -1) for f in features], dtype=torch.long)

        # Fill tensors with data
        for i, feature in enumerate(features):
            seq_len = len(feature["input_ids"])

            input_ids[i, :seq_len] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1

            if "labels" in feature and feature["labels"] is not None:
                labels[i, :seq_len] = torch.tensor(feature["labels"], dtype=torch.long)
            else:
                # For causal LM, fallback to input_ids
                labels[i, :seq_len] = torch.tensor(feature["input_ids"], dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "dataset_id": dataset_ids,
        }

        return batch
    


    