import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

x_np=np.linspace(-2,2,200,dtype=np.float32)
x=torch.unsqueeze(torch.from_numpy(x_np),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())


plt.style.use("dark_background")
# plt.scatter(x,y)
# plt.show()

class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=nn.Linear(n_feature,n_hidden)
        self.predict=nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net=Net(1,10,1)

print(net)

plt.ion()
plt.show()

optimizer=torch.optim.SGD(net.parameters(),lr=0.3)
loss_func=nn.MSELoss()

for t in range(200):
    prediction=net(x)

    loss=loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    if t%2==0:
        plt.cla()
        plt.scatter(x,y)
        plt.plot(x.numpy(),prediction.detach().numpy(),'red',lw=5)
        plt.text(0.5,0,'Loss=%.4f' % loss ,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)



plt.ioff()
plt.show()