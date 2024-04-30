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
        self.linear2 = nn.Linear(ff_dim, input_dim)
    def forward(self, x):
        # 保存原始输入
        orig_x = x

        x = self.linear1(x)
        #应用torch.nn.functional模块中封装的GELU函数
        x = F.gelu(x)
        # 通过第二个线性层
        x = self.linear2(x)

        # 添加残差连接（将处理后的x与原始x相加）
        x += orig_x

        return x# 创建前馈网络实例
# 通过前馈神经网络
fnn = FeedForwardNetwork(input_dim=input_dim, ff_dim=ff_dim)
output = fnn(x)
print(output)