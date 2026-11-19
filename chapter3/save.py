import torch
from pathlib import Path
from matplotlib import pyplot as plt

save_dir=Path(__file__).resolve().parent /"module"

def save():
    net=torch.nn.Sequential(
        torch.nn.Linear(x.shape[1],10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,y.shape[-1])
    )

    optimizer=torch.optim.SGD(net.parameters(),lr=0.2)
    loss_func=torch.nn.MSELoss()

    for i in range(200):
        prec=net(x)
        loss=loss_func(prec,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_dir.mkdir(parents=True,exist_ok=True)
    torch.save(net,save_dir/'net.pkl')
    torch.save(net.state_dict(),save_dir/'net_par.pt')

    plt.figure(1,figsize=(10,3))
    plt.subplot(1,3,1)
    plt.title('net')
    plt.scatter(x.detach().numpy(),y)
    plt.plot(x,prec.detach().numpy(),color='red',lw=5)

def restore_net():
    net1=torch.load(save_dir/'net.pkl',weights_only=False)
    prec=net1(x)
    plt.subplot(1,3,2)
    plt.title('net1')
    plt.scatter(x,y)
    plt.plot(x,prec.detach(),color='red',lw=5)

def restore_params():
    net2=torch.nn.Sequential(
        torch.nn.Linear(x.shape[-1],10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,y.shape[-1])
    )

    net2.load_state_dict(torch.load(save_dir/'net_par.pt'))

    prec=net2(x)
    
    plt.subplot(1,3,3)
    plt.title('net2')
    plt.scatter(x,y)
    plt.plot(x,prec.detach(),color='red',lw=5)
    plt.show()

torch.manual_seed(1)

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)

y=x.pow(2)+0.2*torch.rand_like(x)


save()

restore_net()

restore_params()

