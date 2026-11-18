import torch
import torch.nn.functional as F
import torch.nn as nn

n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)
x1=torch.normal(-2*n_data,1)
y0=torch.zeros(100)
y1=torch.ones(100)
y_target=torch.cat((y0,y1),dim=0)
x_cat=torch.cat((x0,x1),dim=0)


net=nn.Sequential(
    nn.Linear(x_cat.shape[1],10),
    # F.relu(x_cat),
    nn.ReLU(),
    nn.Linear(10,y_target.unique().numel()),
)

print(net)
