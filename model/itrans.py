import torch.nn as nn
import torch
from model.classic_model import Encoder, EncoderLayer, AttentionLayer, FullAttention

class iTrans(nn.Module):
    def __init__(self,config):
        super(iTrans, self).__init__()
        self.config = config
        self.embedding = nn.Linear(config.seq_len,config.d_model)
        self.encoder = Encoder(
            EncoderLayer(
                AttentionLayer(
                    FullAttention(out_attention=False,mask=True)
                    ,d_model=config.d_model,n_heads=config.n_heads
                ),
                d_model=config.d_model,d_ff=config.d_ff,ff_type=config.ff_type
            ),
            num_e_layers=config.num_e_layers
        )
        self.out_linear = nn.Linear(config.d_model,config.pred_len)
    def forward(self,x, x_stamp, y, y_stamp):
        if self.config.task == 'forecast':
            out = self.forecast(x, x_stamp, y, y_stamp)
        return out

    def forecast(self, x, x_stamp, y, y_stamp):
        # Stationarization
        mean = x.mean(dim=1, keepdim=True).detach()
        x = x - mean
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True,unbiased=False)+1e-5)
        x /= stdev
        # InvertEmedding
        x = x.permute(0,2,1)
        x = self.embedding(x)
        enc_out,_ = self.encoder(x,x)
        dec_out = self.out_linear(enc_out)
        dec_out = dec_out.permute(0,2,1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.config.pred_len, 1))
        dec_out = dec_out + (mean[:, 0, :].unsqueeze(1).repeat(1, self.config.pred_len, 1))
        return dec_out





        pass

