# the program checks if PyTorch is using the M1 chip (MPS backend).
"""
This module is developed by Suanfamama.

Authors:
- Wei Jiang (wei@suanfamama.com)
- Mama Xiao (mama.xiao@suanfamama.com)
"""

import torch
print(torch.__version__) # 2.1.0

# Step 1: Check PyTorch Installation
# Check if MPS is available
if torch.backends.mps.is_available():
    print("MPS backend is available.")
else:
    print("MPS backend is not available.")

# Step 2: Verify Device Availability
# Check if PyTorch is set to use MPS by default if available
if torch.backends.mps.is_built():
    print("PyTorch is built with MPS support.")
else:
    print("PyTorch is not built with MPS support.")

# Step 3: Run a Simple Computation on MPS
if torch.backends.mps.is_available():
    # Create a tensor on MPS
    mps_device = torch.device("mps")
    x = torch.randn(3, 3, device=mps_device)
    y = torch.randn(3, 3, device=mps_device)
    z = x + y

    print("Tensor computation on MPS device:")
    print(z)
else:
    print("MPS device is not available.")

'''
2.1.0
MPS backend is available.
PyTorch is built with MPS support.
Tensor computation on MPS device:
tensor([[-0.5232, -0.4680,  2.2931],
        [ 1.2223, -1.3687, -0.5546],
        [-2.0788,  2.1823,  0.4195]], device='mps:0')
'''