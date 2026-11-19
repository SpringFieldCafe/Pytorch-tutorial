import torch
import torch.utils.data as Data

BATCH_SIZE=5

def main():
    x=torch.linspace(1,10,10)
    y=torch.linspace(10,1,10)

    tor_dataset=Data.TensorDataset(data_tensor=x,target_tensor=y)
    loader=Data.DataLoader(
        dataset=tor_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    for epoch in range(3):
        for step,(batch_x,batch_y) in enumerate(loader):
            print(f'Epoch:{epoch}',f'| Step:{step}','| batch x:',batch_x,'| batch y: ',batch_y)


if __name__=='__main__':
    main()