#导入PyTorch库及其相关组件
import torch
import torch.nn as nn
import torch.nn.functional as F
#初始化参数并创建输入数据张量
input_dim = 512  # 输入层维度
ff_dim = 2048  # 前馈网络内部层的维度
#输入数据的张量，batch_size为64，序列长度为50
x = torch.rand(64, 50, input_dim)

#前馈神经网络模块
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, ff_dim):
        #初始化前馈神经网络模块
        super(FeedForwardNetwork, self).__init__()
        # 第一个线性层
        self.linear1 = nn.Linear(input_dim, ff_dim)
        # 第二个线性层，其输出维度与输入维度相同
        self.linear2 = nn.Linear(ff_dim, input_dim)
    def forward(self, x):
        #前向传播函数
        # 通过第一个线性层
        x = self.linear1(x)
        #应用torch.nn.functional模块中封装的GELU函数
        x = F.gelu(x)
        # 通过第二个线性层
        x = self.linear2(x)
        return x# 创建前馈网络实例
# 通过前馈神经网络
fnn = FeedForwardNetwork(input_dim=input_dim, ff_dim=ff_dim)
output = fnn(x)
print(output)