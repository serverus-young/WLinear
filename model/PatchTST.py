import torch.nn as nn
import torch
from layers.embedding import PatchEmbedding
from model.classic_model import Encoder, EncoderLayer, AttentionLayer, FullAttention,Decoder, DecoderLayer
class PatchTST(nn.Module):
    def __init__(self, configs):
        super(PatchTST, self).__init__()
        self.configs = configs
        self.stride = 8
        self.embedding = PatchEmbedding(patch_size=configs.patch_size, d_model=configs.d_model, stride=8, padding=8)
        self.encoder = Encoder(
            EncoderLayer(
                AttentionLayer(
                    FullAttention(out_attention=True, mask=True),
                    d_model= configs.d_model, n_heads= configs.n_heads
                ),
                d_model=configs.d_model, d_ff= configs.d_ff, ff_type= configs.ff_type, dropout= 0.1
            ),
            num_e_layers= 2
        )
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - configs.patch_size) /self.stride + 2)
        self.flatten = Flatten(self.head_nf, configs.pred_len)
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
        x = self.embedding(x)
        x, att_weight= self.encoder(x,x)
        x = x.reshape(-1, var, x.shape[-2], x.shape[-1])

        out = x.permute(0, 1, 3, 2)
        out = self.flatten(out)
        out = out.permute(0,2,1)

        out = out * (stdev[:,0,:].unsqueeze(1).repeat(1, self.configs.pred_len,1))
        out = out + (mean[:,0,:].unsqueeze(1).repeat(1, self.configs.pred_len,1))

        return out

class Flatten(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        super(Flatten, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x



