from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
# 通过设备网格根据给定的 world_size 创建分片计划
device_mesh = DeviceMesh("cuda", torch.arange(0, args.world_size))
#DeviceMesh用于表示设备的拓扑结构。使用 torch.arange创建一个包含 args.world_size个 GPU 的设备网格
# 创建模型并移动到GPU
model = ToyModel().cuda(rank)
# 为并行化模块创建优化器
#初始化SGD优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
# 张量并行化模块，使用 parallelize_module函数对模型进行并行化
# 指定PairwiseParallel，将列向和行向并行样式串联，一种独立的计算策略，
model = parallelize_module(model, device_mesh, PairwiseParallel())
# 对分片模块执行多次前向/后向传播和优化器对参数进行更新。
for i in range(args.iter_nums):
    # 对于 TP，所有 TP rank 的输入需要相同。
    # 设置随机种子是为了模仿数据加载器的行为。
    if rank == 0:
        print(f"-----------{i}--------------")
    torch.manual_seed(i)
    inp = torch.rand(20, 10).cuda(rank)
    if rank == 0:
        print(f"rank: {rank} , input shape: {inp.shape}")
    output = model(inp)
    if rank == 0:
        print(f"rank: {rank} , output shape: {output.shape}")
    output.sum().backward()
    optimizer.step()