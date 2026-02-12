import torch

print("\n--- GPU INFO ---")
if torch.cuda.is_available():
    try:
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    except:
        print("GPU 0: Not found or Error")
        
    try:
        print(f"GPU 1: {torch.cuda.get_device_name(1)}")
    except:
        print("GPU 1: Not found or Error")
else:
    print("No CUDA devices found.")