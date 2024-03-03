import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    for i in range(num_gpus):
        gpu = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu.name}, Memory: {gpu.total_memory / (1024 ** 3)}GB")
else:
    print("CUDA is not available.")


# cpu test

device = torch.device("cpu")
x = torch.randn(3, 3).to(device)
