import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def set_seed(seed):
                       # Python 内置随机模块
    np.random.seed(seed)                 # NumPy
    torch.manual_seed(seed)             # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)     # 当前 GPU
        torch.cuda.manual_seed_all(seed) # 所有 GPU
        torch.backends.cudnn.deterministic = True  # 可选：确保确定性
        torch.backends.cudnn.benchmark = False     # 可选：禁用优化