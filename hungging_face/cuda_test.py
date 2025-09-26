import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Torchvision:", torchvision.__version__)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
