import numpy as np
def cross_entropy_loss(y_true, y_pred):
    #y_true: 真实的标签，one-hot 编码的向量
    #y_pred: 预测的概率分布，每个类别的预测概率
    # 确保不会计算log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    # 计算交叉熵
    ce_loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return ce_loss