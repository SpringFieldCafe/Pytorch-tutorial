import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path

EPOCH=1
BATCH_SIZE=64
TIME_STEP=28
INPUR_SIZE=28
LR=0.01
DOWNLOAD_MNIST=False

save_dir=Path(__file__).resolve().parent /'MNIST'


