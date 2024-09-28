import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# 定义你的自定义数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, input_size, data_size):
        self.data = torch.randn(data_size, input_size)
        self.labels = torch.randn(data_size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# 定义你的模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def main(rank, world_size, args):
    # 1. 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    # 2. 配置每个进程的 GPU
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # 3. 创建分布式并行模型
    model = MyModel().to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # 4. 为数据集创建取样器
    dataset = MyDataset(input_size=10, data_size=1000)
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    trainloader=DataLoader(dataset=dataset,sampler=train_sampler,batch_size=args.batch_size,pin_memory=True, num_workers=args.workers)

    # 5. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练循环
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        trainloader.sampler.set_epoch(epoch)  # 设置 epoch，以便数据在每个 epoch 进行不同的划分
        for data, label in trainloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            prediction = model(data)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

        if rank == 0:  # 只在主进程打印
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 6. 销毁进程组
    dist.destroy_process_group()
#设置命令行参数，获取 GPU 数量，并使用 torch.multiprocessing.spawn 启动多个进程来进行分布式训练。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)