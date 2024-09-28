import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl')

# 获取当前进程的GPU设备号
local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)



def load_data(text, batch_size, max_seq_len, is_train=True):
    dataset = DataSet(text, max_seq_len)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_train else None
    return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=(sampler is None))



# 初始化模型
model = LLaMA2(tokenizer.getVocabSize(), max_seq_len, 256, 16, 4, 8).to(local_rank)

# 使用 DDP 包装模型
model = DDP(model, device_ids=[local_rank], output_device=local_rank)


def train_model(epoch, model, optimizer, loss_fn, train_data_iter, test_data_iter, device):
    test_perplexity = []
    for i in range(epoch):
        train_loss = []
        # 设置分布式采样器的 epoch，保证数据分割不同
        train_data_iter.sampler.set_epoch(i)

        # train
        for k, (idx, target) in enumerate(train_data_iter):
            B, T = idx.shape
            idx = idx.to(device)
            target = target.to(device)
            model = model.to(device)
            x, logits = model(idx)  # 模型预测
            x = x.view(B * T, -1)
            target = target.view(B * T)
            loss = loss_fn(x, target)

            optimizer.zero_grad(set_to_none=True)  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 完成一轮训练，更新参数
            train_loss.append(loss.item())
            if k % 1000 == 0 and dist.get_rank() == 0:  # 仅在主进程上打印信息
                model.eval()
                print(f"step {k} loss :{sum(train_loss) / len(train_loss)}")
                print(tokenizer.decode(
                    generate(model.module,  # DDP 包装的模型需要通过 module 访问原模型
                             torch.tensor(tokenizer.encode("昭通机场", bos=False, eos=False), dtype=torch.long,
                                          device=device).unsqueeze(0),
                             max_new_tokens=200)[0].to('cpu').tolist()))
                model.train()
        if dist.get_rank() == 0:  # 仅在主进程上打印信息
            print(f"epoch-{i} train loss: {sum(train_loss) / len(train_loss)}")
            perplexity = compute_perplexity(model, test_data_iter, device)
            test_perplexity.append(perplexity)
            print(f"epoch-{i} test perplexity: {perplexity}")
    if dist.get_rank() == 0:  # 仅在主进程上绘制图表
        plot_perplexity(test_perplexity)