import torch
import torch.nn as nn
from .blocks import FastMultiHeadAttention, FeedForward

class Encoder(nn.Module):
    # TODO INIT METHOD
    def __init__(self, model_dim: int, num_heads: int, dropout: float, ff_dim: int, swiglu: bool = False):
        super(Encoder, self).__init__()

        self.s_atten = FastMultiHeadAttention(model_dim, num_heads)
        self.layer_1_norm = nn.LayerNorm(model_dim)
        self.feed_fwrd = FeedForward(model_dim, model_dim, ff_dim, dropout, swiglu)
        self.layer_2_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask, is_causal=False):
        s_atten_ouptut = self.s_atten(x, x, x, mask, is_causal)
        res_normed = self.layer_1_norm(self.dropout(s_atten_ouptut) + x)
        feed_fwrd = self.feed_fwrd(res_normed)
        encoder_output = self.layer_2_norm(res_normed + self.dropout(feed_fwrd))
        return encoder_output