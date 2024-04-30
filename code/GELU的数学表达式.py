import numpy as n
def gelu(x):
    #参数x: 一个numpy数组或一个标量
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))