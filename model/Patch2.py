import torch.nn as nn
import torch
from layers.embedding import PatchEmbedding, PositionalEmbedding, Patch2Embedding
from model.classic_model import Encoder, EncoderLayer, AttentionLayer, FullAttention
class Patch2(nn.Module):
    def __init__(self, configs):
        super(Patch2, self).__init__()
        self.configs = configs
        self.stride = 8
        self.embedding = Patch2Embedding(patch_size=configs.patch_size, d_model=configs.d_model, stride=8, padding=8)
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
        self.out_linear = nn.Linear(configs.d_model, self.configs.pred_len)
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - configs.patch_size) /self.stride + 2)
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
        # （16,96,7）->(112,12,16)->(112,12,512)
        x_k = self.embedding(x)
        x = x.permute(0, 2, 1)
        x_q = self.linear(x) + self.positional_embedding(x)
        x, att= self.encoder(x_q,x_k)
        x = self.out_linear(x)

        out = x.permute(0,2,1)

        out = out * (stdev[:,0,:].unsqueeze(1).repeat(1, self.configs.pred_len,1))
        out = out + (mean[:,0,:].unsqueeze(1).repeat(1, self.configs.pred_len,1))

        return out,att

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



