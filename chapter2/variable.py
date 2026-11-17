import torch
from torch.autograd import Variable

tensor2=torch.FloatTensor([[4,6],[2,9]])
vr=Variable(tensor2,requires_grad=True)   

print("\nvr:",vr,)

t_out=torch.mean(tensor2*tensor2)
v_out=torch.mean(vr*vr)

print(t_out,'\n',v_out)

v_out.backward()
print('\n',vr.grad,'\n')

tensor=torch.FloatTensor([[2,5],[-3,-1]])
tensor1=torch.tensor([[-3.0,1],[-2.5,7.8]],requires_grad=True)
t_out=torch.mean(tensor1*tensor1)

print('\n',tensor1.data)
