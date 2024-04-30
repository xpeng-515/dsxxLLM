import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True,	transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64,	shuffle=True)

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, ff_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.dropout = nn.Dropout(0.2) # 添加Dropout层
        self.linear2 = nn.Linear(ff_dim, input_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x) # 在GELU后应用Dropout
        x = self.linear2(x)
        return x

# 实例化模型、定义损失函数和优化器
model = FeedForwardNetwork(28*28, 512) # 假设ff_dim为512
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,	weight_decay=1e-4) # 添加L2正则化（权重衰减）
# 训练模型
epochs = 5
for epoch in range(epochs):
    model.train() # 确保模型处于训练模式
    for data, target in train_loader:
        data = data.view(data.size(0), -1) # 展平图像
        optimizer.zero_grad() # 清零梯度
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training complete.")

#输出示例
#Epoch 1, Loss: 0.3179131746292114
#Epoch 2, Loss: 0.07895921170711517
#Epoch 3, Loss: 0.14426617324352264
#Epoch 4, Loss: 0.2378133237361908
#Epoch 5, Loss: 0.06409002095460892
#Training complete.