import torch
import tensorrt

print(torch.cuda.is_available())        # Should return True
print(torch.cuda.get_device_name(0))    # Show GPU name
print(tensorrt.__version__)             # Show TensorRT version