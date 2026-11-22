import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from pathlib import Path
import os


DOWNLOAD_MNIST=True

save_dir=Path(__file__).resolve().parent /"MNIST"

MNIST=torchvision.datasets.MNIST
MNIST.mirrors=["https://azureopendatastorage.blob.core.windows.net/mnist/",*MNIST.mirrors,]

os.environ["HTTPS_PROXY"]="http://127.0.0.1:7897"


