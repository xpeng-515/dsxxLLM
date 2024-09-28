import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 数据加载和预处理
transform = transforms.Compose([transforms.ToTensor(), 	transforms.Normalize((0.5,), (0.5,))])
trainset = MNIST(root='./data', train=True, download=True, 	transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, 	shuffle=True)
# 定义前馈神经网络模型
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, ff_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x
# 因为MNIST是28x28的图片，我们需要将其展平为784维向量
model = FeedForwardNetwork(28*28, 512).to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(-1, 28*28)  # 展平图片
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()  # 更新学习率
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training complete")
##输出示例
#Epoch 1, Loss: 0.12171544134616852
#Epoch 2, Loss: 0.02324264869093895
#Epoch 3, Loss: 0.012344680726528168
#Epoch 4, Loss: 0.08576418459415436
#Epoch 5, Loss: 0.14215686917304993
#Training complete