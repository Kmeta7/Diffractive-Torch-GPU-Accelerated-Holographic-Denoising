import torch

print("PyTorch version:", torch.__version__)

# CUDA / ROCm availability
print("CUDA available:", torch.cuda.is_available())
print("ROCm available:", torch.version.hip is not None)

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Number of GPUs: {n}")

    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}:")
        print("  Name:", props.name)
        print("  Total memory (GB):", round(props.total_memory / 1024**3, 2))
        print("  Compute capability:", getattr(props, "major", "?"), ".", getattr(props, "minor", "?"))

    # current device
    print("\nCurrent device index:", torch.cuda.current_device())
    print("Current device name:", torch.cuda.get_device_name())

else:
    print("No GPU detected by torch.cuda")
