import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

TIME_STEP=10
INPUT_SIZE=1
LR=0.02

# steps=np.linspace(0,np.pi*2,100,dtype=np.float32)
# x_np=np.sin(steps)
# y_np=np.cos(steps)

# plt.plot(steps,x_np,'red',label="input(sin)")
# plt.plot(steps,y_np,'blue',label="output(cos)")
# plt.legend(loc='best')
# plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out=nn.Linear(32,1)

    def forward(self,x,h_state):
        r_out,h_state=self.rnn(x,h_state)
        outs=[]
