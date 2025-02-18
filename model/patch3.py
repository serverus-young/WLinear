import torch.nn as nn
import torch
from layers.embedding import PatchEmbedding, PositionalEmbedding
from model.classic_model import Encoder, EncoderLayer, AttentionLayer, FullAttention
class Patch3(nn.Module):
    def __init__(self, configs):
        super(Patch3, self).__init__()
        self.configs = configs
        self.stride = 8
        self.patch_num = int((configs.seq_len - configs.patch_size) /self.stride + 2)
        self.embedding = Patch3Embedding(self.patch_num,configs.e_in,patch_size=configs.patch_size, d_model=configs.d_model, stride=8, padding=8)
        self.encoder = Encoder(
            EncoderLayer(
                AttentionLayer(
                    FullAttention(out_attention=False, mask=True),
                    d_model= configs.d_model, n_heads= configs.n_heads
                ),
                d_model=configs.d_model, d_ff= configs.d_ff, ff_type= configs.ff_type, dropout= 0.1
            ),
            num_e_layers= 2
        )
        self.positional_embedding = PositionalEmbedding(configs.d_model)
        self.linear = nn.Linear(96, configs.d_model)
        self.out_linear = nn.Linear(configs.d_model, configs.patch_size)
        self.head_nf = configs.d_model *  configs.patch_size
        self.flatten = Flatten2(self.head_nf, configs.pred_len)
    def forward(self,x, x_stamp, y, y_stamp):
        if self.configs.task == 'forecast':
            out = self.forecast(x, x_stamp, y, y_stamp)
        return out

    def forecast(self, x, x_stamp, y, y_stamp):
        # Non-Stationary Transformer normalization
        var = x.shape[-1]
        mean = x.mean(dim=1, keepdim=True).detach()
        x = x - mean
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        x /= stdev
        # （16,96,7）->(16×16,7,512),(16×16,12,512)
        x_q,x_k = self.embedding(x)
        x, _= self.encoder(x_q,x_k)
        x = x.reshape(-1, x.shape[1], self.configs.patch_size, x.shape[-1])


        x = x.permute(0,1,3,2)
        x = self.flatten(x)
        x = x.permute(0,2,1)
        out = x * (stdev[:,0,:].unsqueeze(1).repeat(1, self.configs.pred_len,1))
        out = out + (mean[:,0,:].unsqueeze(1).repeat(1, self.configs.pred_len,1))

        return out

class Flatten2(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        super(Flatten2, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Patch3Embedding(nn.Module):
    def __init__(self, patch_num,e_in,patch_size, stride, padding, d_model, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = nn.ReplicationPad1d((0, padding))
        self.linear_q= nn.Linear(patch_num, d_model)
        self.linear_k = nn.Linear(e_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        # （16,96,7）->(16×16,7,512),(16×16,12,512)
        x = x.permute(0,2,1)
        x = self.padding(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # （16,12,512）
        x_q = x.reshape(-1,x.shape[1],x.shape[2])
        x_k = x.reshape(-1,x.shape[2],x.shape[1])
        x_q = self.linear_q(x_q) + self.position_embedding(x_q)
        x_k = self.linear_k(x_k) + self.position_embedding(x_k)
        return self.dropout(x_q), self.dropout(x_k)



