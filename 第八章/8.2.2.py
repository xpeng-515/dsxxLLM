import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe
from torch import distributed as dist

# 初始化进程组，使用 NCCL 后端，以支持 GPU 间的高效通信
dist.init_process_group(backend='nccl')

# 创建包含两个线性层的顺序模型，并将每个层分配到不同的 GPU 上（cuda:0 和 cuda:1）。
fc1 = nn.Linear(16, 8).to('cuda:0')
fc2 = nn.Linear(8, 4).to('cuda:1')
model = nn.Sequential(fc1, fc2)

# 使用 Pipe 进行流水线并行，设置微批次micro-batches 为 8
model = Pipe(model, chunks=8)

# 准备输入数据，并将其分配到第一个 GPU
input = torch.rand(16, 16).to('cuda:0')

# 前向传播
output = model(input)