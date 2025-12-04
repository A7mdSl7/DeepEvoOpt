import torch
import sys

def verify_gpu():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Device Count: {device_count}")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
        # Test tensor creation
        try:
            x = torch.tensor([1.0, 2.0]).cuda()
            print("Successfully created tensor on GPU.")
        except Exception as e:
            print(f"Failed to create tensor on GPU: {e}")
    else:
        print("Using CPU. Please check your Colab runtime settings (Runtime > Change runtime type > Hardware accelerator > GPU).")

if __name__ == "__main__":
    verify_gpu()
