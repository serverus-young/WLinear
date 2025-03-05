import torch
import torch.nn as nn
from pytorch_wavelets import DWT1D, IDWT1D  # 一维离散小波变换
from matplotlib import pyplot as plt
from torch.nn import MSELoss

from utils.metric  import metric
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class xiaorong(nn.Module):
    def __init__(self, args):
        super(xiaorong, self).__init__()
        self.configs = args
        # 配置 DWT1D 和 IDWT1D
        self.fc1 = nn.Linear(args.seq_len, args.d_model)
        self.proj = nn.Linear(args.d_model, args.pred_len )
        # self.wavecore = Waveletcore(args.seq_len//2 + 2, args.pred_len//2 + 2)

    def forward(self, x, x_stamp, y, y_stamp):
        mean = x.mean(dim=1, keepdim=True)  # 按时间维度求均值
        x = x - mean
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        x /= stdev

        pred = self.fc1(x.permute(0,2,1))
        pred = self.proj(pred).permute(0,2,1)

        pred = pred * stdev + mean
        return pred
