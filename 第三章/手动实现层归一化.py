import numpy as np
def layer_normalization(inputs, epsilon=1e-5):
    #inputs: 输入数据，形状为(batch_size, num_features)。
    #epsilon: 用于防止除以零的小常数。
    # 计算每个样本的均值和方差
    mean = np.mean(inputs, axis=1, keepdims=True)
    var = np.var(inputs, axis=1, keepdims=True)
    # 归一化
    normalized = (inputs - mean) / np.sqrt(var + epsilon)
    return normalized