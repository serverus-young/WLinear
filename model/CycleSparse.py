import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model.classic_model import Encoder, EncoderLayer, AttentionLayer, FullAttention
from layers.Embed import DataEmbedding
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
class CycleSparse(nn.Module):
    def __init__(self,configs,periods):
        super(CycleSparse, self).__init__()
        self.config = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.period_len = 48
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)
        linear_layer = []
        for i in range(len(periods)):
            period = periods[i]
            if (self.config.seq_len) % period != 0:
                a = self.config.seq_len % period
                b  = period - ((a + configs.pred_len) % period)
                linear = nn.Linear((configs.seq_len-a)// period, (configs.pred_len+ a + b)//period)
            else:
                linear = nn.Linear((configs.seq_len // period), configs.pred_len//period)
            linear_layer.append(linear)
        self.linear_layers = nn.ModuleList(linear_layer)
    def forward(self,x,periods):
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x1 = (x - seq_mean).permute(0, 2, 1)
        B, C, L = x1.size()
        pred_list = []
        for i in range(len(periods)):
            period = periods[i]
            # padding
            if (self.seq_len) % period != 0:
                a = self.seq_len % period
                x = x1[:,:,:self.config.seq_len-a]
                length = self.config.seq_len - a
                x = x.reshape(-1, length // period, period).permute(0, 2, 1)
                y = self.linear_layers[i](x)
                y = y.permute(0, 2, 1).reshape(B, C, -1)
                y = y[:,:,a: self.config.pred_len + a].permute(0, 2, 1)
            else:
                x = x1.reshape(-1, self.seq_len//period, period).permute(0, 2, 1)
                y = self.linear_layers[i](x)
                y = y.permute(0,2,1).reshape(B,C,-1).permute(0,2,1)
            pred_list.append(y)
        pred = torch.stack(pred_list,dim=-1)
        y= torch.mean(pred,dim=-1)
        y = y + seq_mean
        return y