import torch

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))
print("torch cuda version:", torch.version.cuda)
