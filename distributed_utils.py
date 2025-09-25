"""
Distributed Training Utilities for SFT Training Pipeline
Handles distributed training setup, synchronization, and cleanup
"""

import os
import logging
import torch
import torch.distributed as dist
from typing import Tuple, Optional
import subprocess
import socket
from contextlib import contextmanager


def setup_distributed(backend: str = "nccl") -> Tuple[int, int]:
    """
    Setup distributed training environment

    Args:
        backend: Distributed backend ("nccl", "gloo", or "mpi")

    Returns:
        Tuple of (local_rank, world_size)
    """
    logger = logging.getLogger(__name__)

    # Check if we're in a distributed environment
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
    elif "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        # Single GPU training
        logger.info("No distributed environment detected, using single GPU")
        return -1, 1

    logger.info(f"Initializing distributed training: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    # Initialize process group
    try:
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=world_size,
                rank=rank
            )

        # Verify initialization
        if dist.is_initialized():
            logger.info(f"Distributed training initialized successfully with {backend} backend")
            logger.info(f"Process {rank}/{world_size} on device {local_rank}")
        else:
            raise RuntimeError("Failed to initialize distributed training")

    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        raise

    return local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.getLogger(__name__).info("Distributed training cleaned up")


def get_rank() -> int:
    """Get current process rank"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Get local rank"""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    return 0


def is_main_process() -> bool:
    """Check if current process is the main process"""
    return get_rank() == 0


def barrier():
    """Synchronize all processes"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@contextmanager
def main_process_first():
    """Context manager to run code on main process first, then others"""
    if not is_main_process():
        barrier()

    yield

    if is_main_process():
        barrier()


def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """Reduce tensor across all processes"""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    # Clone to avoid modifying original tensor
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=op)

    return reduced_tensor


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensor from all processes"""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    world_size = get_world_size()

    # Create list to hold gathered tensors
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    # Gather tensors
    dist.all_gather(tensor_list, tensor)

    # Concatenate along batch dimension
    return torch.cat(tensor_list, dim=0)


def setup_for_distributed_training(model, device_id: int):
    """Setup model for distributed training"""
    if not (dist.is_available() and dist.is_initialized()):
        return model

    from torch.nn.parallel import DistributedDataParallel as DDP

    # Move model to specific device
    model = model.to(device_id)

    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=False  # Set to True if you have unused parameters
    )

    return model


def create_distributed_sampler(dataset, shuffle: bool = True):
    """Create distributed sampler for dataset"""
    if not (dist.is_available() and dist.is_initialized()):
        return None

    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle
    )


def save_on_master(obj, path: str):
    """Save object only on master process"""
    if is_main_process():
        torch.save(obj, path)


def load_checkpoint_for_distributed(checkpoint_path: str, model, optimizer=None, lr_scheduler=None):
    """Load checkpoint in distributed setting"""
    logger = logging.getLogger(__name__)

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0

    # Load on CPU first to avoid GPU memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state dict
    if hasattr(model, 'module'):
        # DDP model
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state dict
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state dict
    if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    step = checkpoint.get('step', 0)

    logger.info(f"Loaded checkpoint from {checkpoint_path} at step {step}")
    return step


def save_checkpoint_for_distributed(
        checkpoint_path: str,
        model,
        optimizer=None,
        lr_scheduler=None,
        step: int = 0,
        additional_info: dict = None
):
    """Save checkpoint in distributed setting"""
    if not is_main_process():
        return

    checkpoint = {
        'step': step,
    }

    # Save model state dict
    if hasattr(model, 'module'):
        # DDP model
        checkpoint['model_state_dict'] = model.module.state_dict()
    else:
        checkpoint['model_state_dict'] = model.state_dict()

    # Save optimizer state dict
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # Save scheduler state dict
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()

    # Add additional info
    if additional_info:
        checkpoint.update(additional_info)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    logging.getLogger(__name__).info(f"Saved checkpoint to {checkpoint_path} at step {step}")


def print_distributed_info():
    """Print information about the distributed setup"""
    logger = logging.getLogger(__name__)

    if not (dist.is_available() and dist.is_initialized()):
        logger.info("Distributed training not available or not initialized")
        return

    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

    logger.info(f"Distributed Training Info:")
    logger.info(f"  - Rank: {rank}")
    logger.info(f"  - World Size: {world_size}")
    logger.info(f"  - Local Rank: {local_rank}")
    logger.info(f"  - Backend: {dist.get_backend()}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        logger.info(f"CUDA Info:")
        logger.info(f"  - Device Count: {device_count}")
        logger.info(f"  - Current Device: {current_device}")
        logger.info(f"  - Device Name: {device_name}")


def launch_distributed_training(script_path: str, config_path: str, num_gpus: int, **kwargs):
    """Launch distributed training using torchrun"""
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=localhost",
        "--master_port=12355",
        script_path,
        "--config_file", config_path
    ]

    # Add additional arguments
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])

    logging.getLogger(__name__).info(f"Launching distributed training: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.getLogger(__name__).error(f"Distributed training failed: {e}")
        raise


def find_free_port() -> int:
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed_env(
        master_addr: str = "localhost",
        master_port: Optional[int] = None,
        backend: str = "nccl"
):
    """Setup distributed environment variables"""
    if master_port is None:
        master_port = find_free_port()

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # Set backend
    os.environ["NCCL_BACKEND"] = backend if backend == "nccl" else "gloo"

    logging.getLogger(__name__).info(f"Distributed environment setup: {master_addr}:{master_port}, backend={backend}")


class DistributedMetrics:
    """Helper class for aggregating metrics across processes"""

    def __init__(self):
        self.metrics = {}

    def update(self, metrics_dict: dict):
        """Update metrics"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def aggregate(self) -> dict:
        """Aggregate metrics across all processes"""
        if not (dist.is_available() and dist.is_initialized()):
            # Single process - just return averages
            return {key: sum(values) / len(values) for key, values in self.metrics.items()}

        aggregated = {}

        for key, values in self.metrics.items():
            # Convert to tensor and reduce across processes
            if values:
                avg_value = sum(values) / len(values)
                tensor = torch.tensor(avg_value, dtype=torch.float32)

                # All-reduce to get sum across processes
                reduced_tensor = reduce_tensor(tensor, op=dist.ReduceOp.SUM)

                # Average across processes
                aggregated[key] = (reduced_tensor / get_world_size()).item()
            else:
                aggregated[key] = 0.0

        return aggregated

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()