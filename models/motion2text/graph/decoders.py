import torch
import torch.nn as nn
from .blocks import SpatioTemporalAttention
from models.motion2text.transformers.blocks import FeedForward


class TDecoder(nn.Module):
    """
    Temporal Transformer Decoder for motion data.
    """
    def __init__(self, model_dim: int, num_heads: int, dropout: float, ff_dim: int, swiglu: bool=False):
        super(TDecoder, self).__init__()
        self.t_atten = SpatioTemporalAttention(model_dim, num_heads, dropout, mode="temporal")
        self.layer_1_norm = nn.LayerNorm(model_dim)

        self.cross_atten = SpatioTemporalAttention(model_dim, num_heads, dropout, mode="temporal")
        self.layer_2_norm = nn.LayerNorm(model_dim)
        
        self.ff = FeedForward(model_dim, model_dim, ff_dim, dropout, swiglu=swiglu)
        self.layer_3_norm = nn.LayerNorm(model_dim)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, encoder_output, encoder_mask, decoder_mask, is_causal=False):
        t_atten_ouptut = self.t_atten(x, x, x, decoder_mask, is_causal)
        res_norm = self.layer_1_norm(x + self.dropout(t_atten_ouptut))

        cross_atten_output = self.cross_atten(res_norm, encoder_output, encoder_output, encoder_mask)
        res_norm = self.layer_2_norm(res_norm + self.dropout(cross_atten_output))

        feed_forward = self.ff(res_norm)
        decoder_output = self.layer_3_norm(res_norm + self.dropout(feed_forward))

        return decoder_output