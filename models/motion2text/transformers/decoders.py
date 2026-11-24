import torch
import torch.nn as nn
from .blocks import FastMultiHeadAttention, FeedForward

class Decoder(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float, ff_dim: int, swiglu: bool = False):
        super(Decoder, self).__init__()

        self.s_attn = FastMultiHeadAttention(model_dim, num_heads)
        self.layer_1_norm = nn.LayerNorm(model_dim)

        self.cross_atten = FastMultiHeadAttention(model_dim, num_heads)
        self.layer_2_norm = nn.LayerNorm(model_dim)

        self.feed_fwrd = FeedForward(model_dim, model_dim, ff_dim, dropout, swiglu)
        self.layer_3_norm = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        """
        Params
        -------
        x : input to the decoder layer
        encoder_output : output from the corresponding encoder (from cross-attention step)
        encoder_mask : mask to ignore certain parts of the encoder's output
        decoder_mask : mask to ignore certain parts of the decoder's input

        Returns
        -------
        returns decoder's output
        """
        atten_output = self.s_attn(x, x, x, decoder_mask)
        x = self.layer_1_norm(self.dropout(atten_output) + x)

        atten_output = self.cross_atten(x, encoder_output, encoder_output, encoder_mask)
        x = self.layer_2_norm(self.dropout(atten_output) + x)

        ff_output = self.feed_fwrd(x)
        x = self.layer_3_norm(self.dropout(ff_output))
        return x