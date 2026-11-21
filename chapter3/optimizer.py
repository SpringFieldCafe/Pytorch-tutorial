import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn.functional as F

#h
LR=0.01
BATCH_SIZE=32
EPOCH=12

x=torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y=x.pow(2)+0.2*torch.normal(torch.zeros(*x.size()))

torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=1,num_workers=2)


def make_net():
    return torch.nn.Sequential(
        torch.nn.Linear(x.shape[-1],20),
        torch.nn.ReLU(),
        torch.nn.Linear(20,y.shape[-1])

    )


net_Adam=make_net()
net_SGD=make_net()
net_RMS=make_net()
net_Mon=make_net()

nets=[net_Adam,net_SGD,net_RMS,net_Mon]

opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
opt_SGD=torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_RMS=torch.optim.RMSprop(net_RMS.parameters(),lr=LR,alpha=0.9)
opt_Mon=torch.optim.SGD(net_Mon.parameters(),lr=LR,momentum=0.8)

optimizers=[opt_Adam,opt_SGD,opt_RMS,opt_Mon]

Lost_func=torch.nn.MSELoss()


def main():
    for epoch in range(EPOCH):
        print(f"epoch:  {epoch}   \n")
        for step,(batch_x,batch_y) in enumerate(loader):
            for net,opt in zip(nets,optimizers):
                output=net(batch_x)
                loss=Lost_func(output,batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()


if __name__=='__main__':
    main()