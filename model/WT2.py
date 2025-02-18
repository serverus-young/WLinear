import torch
import torch.nn as nn
from pytorch_wavelets import DWT1D, IDWT1D  # 一维离散小波变换

class WT2(nn.Module):
    def __init__(self, args):
        super(WT2, self).__init__()
        self.configs = args

        # 配置 DWT1D 和 IDWT1D
        self.cwt = DWT1D(wave='db3', mode='zero')  # 一维离散小波正变换
        self.icwt = IDWT1D(wave='db3', mode='zero')  # 一维离散小波逆变换
        self.yh_linear = nn.Linear(args.seq_len//2 + 2, args.pred_len//2 + 2)
        self.yl_linear = nn.Linear(args.seq_len//2 + 2, args.pred_len//2 + 2)

    def forward(self, x, x_stamp, y, y_stamp, mask_prob=0.1):
        """
        前向传播
        """
        # x 的维度为 (b, l, n)
        b, l, n = x.size()

        # 数据标准化
        mean = x.mean(dim=1, keepdim=True)  # 按时间维度求均值
        x = x - mean
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        x /= stdev

        # 调整输入形状为 (b, n, l) 以适配 DWT1D
        x = x.permute(0, 2, 1)  # 将形状从 (b, l, n) 变为 (b, n, l)

        # 小波正变换
        Yl, Yh = self.cwt(x)  # 返回低频分量 (Yl) 和高频分量 (Yh)

        # 展平高频分量并输入线性层
        yh = Yh[0]
        yh_pred = self.yh_linear(yh)
        Yl_pred = self.yl_linear(Yl)
        # 逆变换
        Yh_pred = []
        Yh_pred.append(yh_pred)
        pred = self.icwt((Yl_pred, Yh_pred))  # 逆变换后的形状为 (b, n, l)

        # 恢复形状为 (b, l, n)
        pred = pred.permute(0, 2, 1)  # 变为 (b, self.pred_len, n)

        # 还原标准化
        pred = pred * stdev + mean

        return pred
