import torch.nn as nn
import torch


class padding_avg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size)
    def forward(self, x):
        front = x[:,0:1,:].repeat(1,(self.kernel_size-1)//2,1)
        back = x[:,-1:,:].repeat(1,(self.kernel_size-1)//2,1)
        torch.cat([front,x,back],dim=1)
        x = self.avg(x.permute(0,2,1)).permute(0,2,1)
        return x
        pass
class decomp_layer(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.padding_avg = padding_avg(kernel_size)
    def forward(self, x):
        trend_init = self.padding_avg(x)
        season_init = x - trend_init
        return trend_init, season_init