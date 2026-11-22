import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from pathlib import Path
import os


EPOCH=1
BATCH_SIZE=50
LR=0.001
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
        self.out=nn.Linear(32*7*7,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return output

def main():
    train_data=torchvision.datasets.MNIST(
        root=save_dir,train=True,transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )

    print(train_data.data.size())
    print(train_data.targets.size())
    # plt.imshow(train_data.train_data[0],cmap='gray')
    # plt.title(f"{train_data.train_labels[0]}")
    # plt.show()

    train_loader=Data.DataLoader(dataset=train_data,shuffle=True,batch_size=BATCH_SIZE,num_workers=2)
    test_data=torchvision.datasets.MNIST(root=save_dir,train=False,transform=torchvision.transforms.ToTensor())
    # test_x=torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:1000]/255.
    # test_y=test_data.targets[:1000]
    test_loader=Data.DataLoader(dataset=test_data,batch_size=1000,shuffle=False)
    test_x,test_y=next(iter(test_loader))

    cnn=CNN()
    print(cnn)

    optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
    loss_func=nn.CrossEntropyLoss()
    
    for epoch in range(EPOCH):
        for step ,(batch_x,batch_y) in enumerate(train_loader):
            output=cnn(batch_x)
            loss=loss_func(output,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if step % 50 ==0:
                with torch.no_grad():
                    test_output=cnn(test_x)
                    pred_y=torch.max(test_output,1)[1].data.squeeze()
                accur=(pred_y==test_y).float().mean().item()
                print(f"EPOCH:| {epoch}      |  train_loss {loss:.4f}    |  test_accur   {accur:.2%}")



if __name__=='__main__':
    main()