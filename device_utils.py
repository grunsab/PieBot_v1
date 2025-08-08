import torch
import platform

def get_optimal_device():
    """
    Automatically detect and return the best available device for PyTorch operations.
    Supports CUDA GPUs (including B200), Apple Silicon MPS, and CPU fallback.
    
    Returns:
        torch.device: The optimal device for computation
        str: String description of the device for logging
    """
    
    # Check for CUDA availability (including B200 and other modern GPUs)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        device_str = f"CUDA GPU: {gpu_name}"
        
        # Print GPU memory info for debugging
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        #print(f"GPU Memory: {total_memory:.1f}GB")
        
        return device, device_str
    
    # Check for Apple Silicon MPS (Metal Performance Shaders)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_str = f"Apple Silicon MPS ({platform.machine()})"
        return device, device_str
    
    # Fallback to CPU
    else:
        device = torch.device('cpu')
        device_str = f"CPU ({platform.processor() or platform.machine()})"
        return device, device_str

def move_to_device(tensor_or_model, device):
    """
    Move tensor or model to the specified device with proper error handling.
    
    Args:
        tensor_or_model: PyTorch tensor or model to move
        device: Target device
        
    Returns:
        Tensor or model moved to the specified device
    """
    try:
        return tensor_or_model.to(device)
    except RuntimeError as e:
        print(f"Warning: Failed to move to {device}, falling back to CPU. Error: {e}")
        return tensor_or_model.to('cpu')

def optimize_for_device(model, device):
    """
    Apply device-specific optimizations to the model.
    
    Args:
        model: PyTorch model
        device: Target device
        
    Returns:
        Optimized model
    """
    model = move_to_device(model, device)
    
    # Enable optimizations for different devices
    if device.type == 'cuda':
        # Enable mixed precision and other CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # For modern GPUs like B200, enable tensor core optimizations
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'backends') and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            
    elif device.type == 'mps':
        # MPS-specific optimizations
        # Ensure model is in float32 for MPS compatibility
        model = model.float()
        
    return model

def get_batch_size_for_device(base_batch_size=1024):
    """
    Adjust batch size based on available device memory.
    
    Args:
        base_batch_size: Default batch size
        
    Returns:
        int: Optimized batch size for the device
    """
    device, _ = get_optimal_device()
    
    if device.type == 'cuda':
        # Get GPU memory and adjust batch size accordingly
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_memory >= 80:  # B200 or similar high-memory GPU
                return base_batch_size * 4
            elif total_memory >= 40:  # High-end GPU
                return base_batch_size * 2
            elif total_memory >= 16:  # Mid-range GPU
                return base_batch_size
            else:  # Lower-end GPU
                return base_batch_size // 2
        except:
            return base_batch_size
            
    elif device.type == 'mps':
        # MPS typically has unified memory
        return base_batch_size
    
    else:  # CPU
        return base_batch_size

def get_num_workers_for_device():
    """
    Get optimal number of workers for data loading based on device and CPU count.
    
    Returns:
        int: Optimal number of workers
    """
    import os
    
    device, _ = get_optimal_device()
    cpu_count = os.cpu_count() or 4
    
    if device.type == 'cuda':
        # For GPU training, use more workers to keep GPU fed
        return min(cpu_count, 24)
    elif device.type == 'mps':
        # MPS benefits from fewer workers due to unified memory
        return cpu_count
    else:  # CPU
        # For CPU training, use fewer workers to avoid overhead
        return min(cpu_count // 2, 4)

def get_gpu_count():
    """
    Get the number of available GPUs.
    
    Returns:
        int: Number of available GPUs
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def get_distributed_batch_size(base_batch_size, world_size):
    """
    Calculate the batch size per process for distributed training.
    
    Args:
        base_batch_size: Total batch size across all processes
        world_size: Number of processes
        
    Returns:
        int: Batch size per process
    """
    return base_batch_size // world_size

def setup_distributed_device(rank):
    """
    Set up device for distributed training.
    
    Args:
        rank: Process rank
        
    Returns:
        torch.device: Device for this process
        str: Device description string
    """
    if torch.cuda.is_available() and rank < torch.cuda.device_count():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        gpu_name = torch.cuda.get_device_name(rank)
        device_str = f"CUDA GPU {rank}: {gpu_name}"
        return device, device_str
    else:
        device = torch.device('cpu')
        device_str = f"CPU (Process {rank})"
        return device, device_str

def get_distributed_backend():
    """
    Get the recommended backend for distributed training based on available hardware.
    
    Returns:
        str: Backend name ('nccl' for GPUs, 'gloo' for CPUs)
    """
    if torch.cuda.is_available():
        return 'nccl'
    return 'gloo'