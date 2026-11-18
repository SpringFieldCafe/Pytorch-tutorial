import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)
y0=torch.zeros(100)
x1=torch.normal(-2*n_data,1)
y1=torch.ones(100)
x=torch.cat((x0,x1),dim=0).type(torch.FloatTensor)
y=torch.cat((y0,y1),dim=0).type(torch.LongTensor)

plt.scatter(x[:,0],x[:,1],c=y,s=100,lw=0,cmap="coolwarm")
plt.ylim((-3,3))
plt.xlim((-3,3))
plt.show()

class Net(nn.Module):
    def __init__(self,n_features,n_hidden,n_outputs):
        super().__init__()
        self.hidden=nn.Linear(n_features,n_hidden)
        self.prediction=nn.Linear(n_hidden,n_outputs)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.prediction(x)
        return x

net_hidden=10
net=Net(x.shape[1],net_hidden,y.unique().numel())

plt.ion()
plt.show()

optimizer=torch.optim.SGD(net.parameters(),lr=0.02)
loss_func=nn.CrossEntropyLoss()

for i in range(100):
    logits=net(x)
    loss=loss_func(logits,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%2==0:
        plt.cla()
        prediction=torch.max(F.softmax(logits,dim=1),1)[1]
        pred_y=prediction.squeeze()
        target_y=y
        plt.scatter(x[:,0],x[:,1],c=pred_y,s=100,lw=0,cmap='coolwarm')
        acc=sum(pred_y==target_y)/200
        plt.text(1.5,-4,'Acc=% .2f'  % acc ,fontdict={'color':'blue','size':6})
        plt.pause(0.1)


plt.ioff()
plt.show()
