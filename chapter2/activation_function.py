import torch 
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np


x=torch.linspace(-5,10,300)
print('\n',type(x),'\n')

x_np=x.numpy()

e=np.linspace(-5,10,300)



y_relu=F.relu(x)
y_sigmoid=F.sigmoid(x)
y_tanh=F.tanh(x)
y_softplus=F.softplus(x)

plt.style.use('dark_background')

plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x,y_relu,c='red',label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x,y_sigmoid,c='red',label='sigmoid')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x,y_tanh,c='red',label='tanh')
plt.ylim((-1.2,1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x,y_softplus,c='red',label='softplus')
plt.ylim((-0.2,6))
plt.legend(loc='best')

plt.show()

k=x.sigmoid()