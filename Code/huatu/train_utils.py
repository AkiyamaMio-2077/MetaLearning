import numpy as np 
import random 
import torch
from torch.backends import cudnn



def accuracy(predictions,tragets):
    #predictions(n,nc), targets(n)
    predictions = predictions.argmax(dim =-1).view(tragets.shape)
    return (predictions == tragets).sum().float()/ tragets.shape[0]



def seed_torch(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"set random seed: {seed}")
    # 下面两项比较重要，搭配使用。以精度换取速度，保证过程可复现。
    # https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = False  # False: 表示禁用
    cudnn.deterministic = True  # True: 每次返回的卷积算法将是确定的，即默认算法。
    # cudnn.enabled = True



