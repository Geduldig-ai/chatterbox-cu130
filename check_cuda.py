import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    
    # Print details for each available device
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
        
    # Create a tensor on the GPU
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    print(f"Tensor on CUDA device: {x}")
    
    # Perform a simple operation on the GPU
    y = x * 2
    print(f"Result of operation on CUDA device: {y}")

else:
    print("CUDA is NOT available. PyTorch will run on CPU.")