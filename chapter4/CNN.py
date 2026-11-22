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


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,

            ),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
