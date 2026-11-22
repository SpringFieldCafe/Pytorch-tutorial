import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from pathlib import Path


DOWNLOAD_MNIST=True

save_dir=Path(__file__).resolve().parent /"MNIST"
