import torch
import torch.nn as nn
import torch.nn.functional as F

input_dim = 512  # 输入层维度
ff_dim = 2048  # 前馈网络内部层的维度
x = torch.rand(64, 50, input_dim)

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, ff_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, ff_dim)
        # 添加第一个层归一化，维度为ff_dim
        self.layer_norm1 = nn.LayerNorm(ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)
        # 第二个层归一化，维度为input_dim
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):  # 确保此方法正确缩进，作为类的一部分
        x = self.linear1(x)
        # 紧跟第一个线性层输出之后且在GELU激活函数前，应用第一个层归一化
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        # 紧跟第二个线性层的输出之后，应用第二个层归一化
        x = self.layer_norm2(x)
        return x

# 创建网络实例并传递数据
fnn = FeedForwardNetwork(input_dim=input_dim, ff_dim=ff_dim)
output = fnn(x)
print(output)
