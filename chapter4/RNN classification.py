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

    def forward(self,x):
        # x=self.rnn(x)
        # x=self.out(x)
        # return x
        r_out,(h_n,h_c)=self.rnn(x,None)
        out=self.out(r_out[:,-1,:])
        return out


def main():

    train_data=dsets.MNIST(root=save_dir,train=True,transform=transforms.ToTensor(),download=DOWNLOAD_MNIST)
    train_loader=torch.utils.data.DataLoader(dataset=train_data,shuffle=True,num_workers=2,batch_size=BATCH_SIZE)

    test_data=dsets.MNIST(root=save_dir,train=False,transform=transforms.ToTensor())
    test_loader=torch.utils.data.DataLoader(dataset=test_data,shuffle=False,batch_size=1000)
    test_x,test_y=next(iter(test_loader))
    test_x=test_x.squeeze(1)

    rnn=RNN()
    print(rnn)

    optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
    loss_func=nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
