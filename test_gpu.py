import torch

gpu_available = torch.cuda.is_available()
print(f"Is the GPU available? {'Yes' if gpu_available else 'No'}")

if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    cuda_version = torch.version.cuda
    print(f"CUDA Version: {cuda_version}")

else:
    print("No GPU detected. Please check your CUDA installation and drivers.")
