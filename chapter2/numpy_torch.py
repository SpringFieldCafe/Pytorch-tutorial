import torch
import numpy as np

np_data=np.arange(6).reshape((2,3))
torch_data=torch.from_numpy(np_data)
tensor2np=torch_data.numpy()

print(
    '\npt:',torch_data,
    '\nnp:',np_data,
    '\nts',tensor2np
)

data=[1,-10,3,-7]
ts=torch.tensor(data)
np_sin=torch_data.sin()
np_mean=np.mean(data)
pt_sin=torch.sin(ts)
pt_abs=torch.abs(ts)


print(
    '\nnp_sin:',np_sin,
    '\nnp_mean:',np_mean,
    '\npt_sin',pt_sin,
    '\npt_abs',pt_abs,
)

dataM=[[4,-3],[-5,-9]]
ts2=torch.FloatTensor(dataM)

print(

    '\n 12:',torch.mean(ts2),
)

print(
    'numpy:',np.matmul(data,data),
    'torch',torch.mm(ts2,ts2)
)