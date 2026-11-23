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



class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM(
            input_size=INPUR_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,

        )
        self.out=nn.Linear(64,10)

