# SFT Training Pipeline for Large Language Models

A comprehensive, production-ready Supervised Fine-Tuning (SFT) pipeline for Large Language Models using HuggingFace Transformers with distributed training support.

## Features

- **Flexible Data Loading**: Support for HuggingFace datasets, local files (JSON, JSONL, CSV, TXT)
- **Distributed Training**: Multi-GPU training with PyTorch DDP
- **Memory Optimization**: Gradient checkpointing, mixed precision (FP16/BF16), quantization support
- **Parameter-Efficient Fine-tuning**: LoRA support for resource-constrained training
- **Advanced Monitoring**: Wandb integration, comprehensive metrics, automatic batch size calculation
- **Production Ready**: Robust error handling, checkpointing, resumable training
- **Modular Design**: Clean, extensible codebase with separate concerns

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd sft-training-pipeline

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Prepare Your Data

The pipeline supports multiple data formats:

#### JSONL Format (Recommended)
```json
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
{"instruction": "Summarize", "input": "Long text...", "output": "Summary..."}
```

#### Chat Format
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

#### Simple Text Format
```json
{"text": "This is a training example with input and expected output."}
```

### 3. Configure Training

Create or modify the configuration file:

```bash
cp sample_config.yaml my_config.yaml
# Edit my_config.yaml according to your needs
```

Key configuration options:
```yaml
# Model
model_name_or_path: "microsoft/DialoGPT-medium"

# Data
dataset_path: "./data"
train_file: "train.jsonl"
validation_file: "validation.jsonl"
max_seq_length: 512

# Training
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 5e-5
num_train_epochs: 3
num_gpus: 2  # Number of GPUs to use

# Optimization
fp16: true
gradient_checkpointing: true

# Optional: LoRA for parameter-efficient training
lora_config:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj"]
```

### 4. Launch Training

#### Single GPU Training
```bash
python sft_main.py --config_file my_config.yaml
```

#### Multi-GPU Training
```bash
# Using the launch script (recommended)
./launch_training.sh --config my_config.yaml --gpus 4

# Or using torchrun directly
torchrun --nproc_per_node=4 sft_main.py --config_file my_config.yaml
```

#### SLURM Cluster
```bash
sbatch slurm_job.sh  # See examples/slurm_job.sh
```

## Architecture Overview

```
sft-training-pipeline/
├── sft_main.py              # Main training script
├── config_manager.py        # Configuration management
├── data_handler.py          # Data loading and preprocessing
├── model_manager.py         # Model and tokenizer management
├── trainer_utils.py         # Custom trainer with enhanced features
├── distributed_utils.py     # Distributed training utilities
├── sample_config.yaml       # Sample configuration file
├── launch_training.sh       # Distributed training launcher
└── requirements.txt         # Python dependencies
```

## Configuration Guide

### Data Configuration

```yaml
# Load from HuggingFace Hub
dataset_name: "squad"
dataset_config_name: "v2.0"

# Or load from local files
dataset_path: "./data"
train_file: "train.jsonl"
validation_file: "val.jsonl"

# Preprocessing
max_seq_length: 512
preprocessing_num_workers: 4
```

### Training Configuration

```yaml
# Batch size and optimization
per_device_train_batch_size: 4      # Adjust based on GPU memory
gradient_accumulation_steps: 2      # Effective batch size = 4 * 2 * num_gpus
learning_rate: 5e-5
weight_decay: 0.01

# Training schedule
num_train_epochs: 3
warmup_steps: 500
lr_scheduler_type: "linear"

# Memory optimization
fp16: true                          # Mixed precision training
gradient_checkpointing: true        # Trade compute for memory
```

### Distributed Training

```yaml
num_gpus: 4                         # Number of GPUs
ddp_backend: "nccl"                 # Communication backend
```

### Parameter-Efficient Fine-tuning

```yaml
lora_config:
  r: 16                             # Rank of adaptation
  lora_alpha: 32                    # LoRA scaling parameter
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"
```

### Quantization (Memory Saving)

```yaml
quantization_config:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true
```

## Advanced Features

### Automatic Batch Size Calculation

The pipeline automatically calculates optimal batch sizes based on:
- GPU memory available
- Model size
- Sequence length
- Target memory utilization

### Comprehensive Metrics

- **Perplexity**: Language model quality metric
- **Token Accuracy**: Token-level prediction accuracy
- **Sequence Accuracy**: Full sequence match accuracy
- **Training Speed**: Samples per second, GPU utilization
- **Memory Usage**: GPU memory allocation and utilization

### Monitoring and Logging

#### Weights & Biases Integration
```yaml
report_to: "wandb"
run_name: "my_sft_experiment"
wandb_project: "llm-fine-tuning"
```

#### TensorBoard Support
```yaml
report_to: "tensorboard"
logging_dir: "./logs"
```

### Checkpointing and Resuming

```yaml
resume_from_checkpoint: "./sft_output/checkpoint-1000"
save_steps: 500
save_total_limit: 3
load_best_model_at_end: true
```

## Data Format Examples

### Instruction-Following Format (Alpaca Style)
```json
{
  "instruction": "Write a haiku about programming",
  "input": "",
  "output": "Code flows like water\nBugs emerge from hidden depths\nDebug, then release"
}
```

### Chat/Conversation Format
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."}
  ]
}
```

### Question-Answer Format
```json
{
  "input": "What is the capital of France?",
  "output": "The capital of France is Paris."
}
```

### Simple Text Completion
```json
{
  "text": "The quick brown fox jumps over the lazy dog."
}
```

## Performance Optimization Tips

### Memory Optimization
1. **Gradient Checkpointing**: Trade compute for memory
   ```yaml
   gradient_checkpointing: true
   ```

2. **Mixed Precision**: Use FP16 or BF16
   ```yaml
   fp16: true  # or bf16: true for newer hardware
   ```

3. **Quantization**: 4-bit or 8-bit quantization
   ```yaml
   quantization_config:
     load_in_4bit: true
   ```

4. **LoRA**: Parameter-efficient fine-tuning
   ```yaml
   lora_config:
     r: 16
     target_modules: ["q_proj", "v_proj"]
   ```

### Training Speed Optimization
1. **Batch Size**: Use largest batch size that fits in memory
2. **Gradient Accumulation**: Simulate larger batches
3. **DataLoader Workers**: Parallel data loading
   ```yaml
   dataloader_num_workers: 4
   preprocessing_num_workers: 8
   ```

### Distributed Training Best Practices
1. **Backend Selection**: Use NCCL for GPU training
2. **Network**: Use high-bandwidth interconnects (InfiniBand)
3. **Data Sharding**: Ensure balanced data distribution

## Troubleshooting

### Common Issues

#### Out of Memory (OOM) Errors
```bash
# Reduce batch size
per_device_train_batch_size: 2

# Enable gradient checkpointing
gradient_checkpointing: true

# Use mixed precision
fp16: true

# Consider quantization
quantization_config:
  load_in_4bit: true
```

#### Distributed Training Issues
```bash
# Check CUDA devices
nvidia-smi

# Verify network connectivity
ping <other_node_ip>

# Check for hanging processes
ps aux | grep python

# Kill hanging processes
pkill -f "python.*sft_main.py"
```

#### Data Loading Issues
```bash
# Check file formats
head -n 1 data/train.jsonl | python -m json.tool

# Validate data
python -c "
import json
with open('data/train.jsonl') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f'Invalid JSON at line {i+1}: {line[:100]}')
        if i > 10:
            break
"
```

### Performance Monitoring

#### System Resources
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU and memory
htop

# Monitor network (for distributed training)
iftop
```

#### Training Progress
```python
# Custom monitoring script
import wandb

# Log custom metrics
wandb.log({
    "custom_metric": value,
    "epoch": epoch,
    "step": step
})
```

## Examples

### Basic Fine-tuning
```yaml
model_name_or_path: "gpt2"
dataset_path: "./data"
train_file: "train.jsonl"
per_device_train_batch_size: 8
num_train_epochs: 3
learning_rate: 5e-5
```

### Large Model with LoRA
```yaml
model_name_or_path: "microsoft/DialoGPT-large"
per_device_train_batch_size: 2
gradient_checkpointing: true
fp16: true
lora_config:
  r: 32
  lora_alpha: 64
  target_modules: ["c_attn", "c_proj"]
```

### Multi-GPU Training
```yaml
num_gpus: 4
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
# Effective batch size: 4 * 4 * 2 = 32
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{sft_training_pipeline,
  title={SFT Training Pipeline for Large Language Models},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/sft-training-pipeline}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Join our Discord community

---

**Happy Fine-tuning! 🚀**