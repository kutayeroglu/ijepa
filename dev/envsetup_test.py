"""
This script is intended for testing whether the dependencies for the project are set up successfully.
It checks the versions of the installed libraries and verifies if CUDA is available for PyTorch.

Dependencies:
- torch
- torchvision
- yaml
- numpy
- cv2 (OpenCV)
- submitit

Usage:
Run this script to print the versions of the installed libraries and check if CUDA is available.

Example:
$ python envsetup_test.py
"""

import torch
import torchvision
import yaml
import numpy
import cv2
import submitit

print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy version: {numpy.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Submitit imported successfully")
