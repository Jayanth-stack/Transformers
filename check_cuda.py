import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("Device Count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i} Name:", torch.cuda.get_device_name(i))
        print(f"Device {i} Capability:", torch.cuda.get_device_capability(i))
else:
    print("CUDA is not available. Using CPU only.")

# Test tensor creation on GPU if available
if torch.cuda.is_available():
    # Create a tensor on GPU
    x = torch.tensor([1, 2, 3]).cuda()
    print("Tensor device:", x.device)
    
    # Check if tensor operations work on GPU
    y = x + x
    print("Operation result device:", y.device)
    print("Operation result:", y)
else:
    print("Cannot create GPU tensors as CUDA is not available.") 