import numpy as np
import torch
import torch.nn as nn
import math

class FastMultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int = 256, num_heads: int = 4):
        super(FastMultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "Model's dimension must be divisible by num_heads"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.qkv_projection = nn.Linear(model_dim, model_dim * 3)
        self.output = nn.Linear(model_dim, model_dim)

    def forward(self, Q, K, V, mask=None, is_causal=False):
        batch_size, seq_len, _ = Q.size()

        # if Q, K, V from Self-Attention
        if Q is K and K is V:
            # [B, S, D] -> [B, S, 3 * D]
            qkv = self.qkv_projection(Q)
            
            # Form back heads [B, S, 3, H, D_head]
            qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            
            # [3, B, H, S, D_head]
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # if Q, K, V from Cross-Attention (Q different from K, V)
            q = F.linear(Q, self.qkv_projection.weight[:self.model_dim], self.qkv_projection.bias[:self.model_dim])
            k = F.linear(K, self.qkv_projection.weight[self.model_dim:self.model_dim*2], self.qkv_projection.bias[self.model_dim:self.model_dim*2])
            v = F.linear(V, self.qkv_projection.weight[self.model_dim*2:], self.qkv_projection.bias[self.model_dim*2:])
            
            q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Use FlashAttention from PyTorch
        # The mask should be a boolean mask (True for keeping, False for masking) or a float mask.
        if mask is not None and len(mask.shape)<=3:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            
            # [B, S, S] -> [B, H, S, S]
            mask = mask[:, None, :]      # [B, 1, S, S]
            mask = mask.expand(-1, self.num_heads, -1, -1)  # [B, H, S, S]

        atten_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask,
            dropout_p=0.2,
            is_causal=is_causal
        )

        # [B, H, S, D_head] -> [B, S, H, D_head] -> [B, S, D]
        atten_output = atten_output.transpose(1, 2).reshape(batch_size, seq_len, self.model_dim)

        return self.output(atten_output)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int = 256, num_heads: int = 4):
        """
        
        """
        super(MultiHeadAttention, self).__init__()

        assert model_dim % num_heads == 0, "Model's dimension must be divisible by num_heads"

        self.model_dim = model_dim  # dimension of the model
        self.num_heads = num_heads  # Number of attention heads
        self.head_dim = model_dim // num_heads  # Dimension of each Query, Key and Value vectors

        self.query = nn.Linear(model_dim, model_dim)  # Query vector
        self.key = nn.Linear(model_dim, model_dim)  # Key vector
        self.value = nn.Linear(model_dim, model_dim)  # Value vector
        self.output = nn.Linear(model_dim, model_dim)  # Output vector

    def scaled_dot_prod(self, Q, K, V, mask=None):
        # Compute attention scores
        atten_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale and Softmax applied to scores
        n = np.sqrt(self.head_dim)
        atten_scores = atten_scores / n

        if mask is not None:
            _MASKING_VALUE = -1e+15 if atten_scores.dtype == torch.float32 else -1e+4
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            atten_scores = atten_scores.masked_fill(mask == 0, _MASKING_VALUE)

        # Attention probabilities
        atten_prob = nn.functional.softmax(atten_scores, dim=-1)

        # Matmul between attention weights and Value
        atten_w = torch.matmul(atten_prob, V)
        return atten_w
    
    def split_heads(self, x):
        # Reshape the input into (batch_size, num_heads, seq_len, head_dim)
        # enables the model to process multiple attention heads and parrallel
        # computation
        batch_size, seq_len, model_dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        # After applying attention to each head, the results are combined into
        # a single tensor of shape (batch_size, seq_len, model_dim)
        batch_size, _, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)

    def forward(self, Q, K, V, mask=None):
        # Apply linear layer
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Compute scaled attention
        atten_output = self.scaled_dot_prod(Q, K, V, mask)
        atten_output = self.combine_heads(atten_output)

        # Combine backs heads and apply final linear layer
        output = self.output(atten_output)
        return output


class PositionalEmbedding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        """
        Compute dynamicalely the positional encoding for input sequences.

        Params
        -------
        model_dim : int
            model's input dimension
        """
        super(PositionalEmbedding, self).__init__()
        self.model_dim = model_dim

        pe = torch.zeros(max_len, 1, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add pre-computed positional encoding to the input tensor x.

        Params
        -------
        x : torch.Tensor
            input tensor of shape (batch_size, seq_len, num_joints, model_dim)
        """
        seq_len = x.size(1)
        x = x.unsqueeze(2) if x.ndim == 3 else x  # Add a dimension for num_joints if not present
        x = x + self.pe[:, :seq_len]
        return x

    # def _compute_pe(self, x):
    #     """
    #     Compute the positional encoding for input tensor x.

    #     Params
    #     -------
    #     x: torch.Tensor
    #         input tensor of shape (batch_size, seq_len, model_dim)
    #     """
    #     seq_len = x.size(1)
    #     device = x.device

    #     position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # Tensor containing the position indices for each position in the sequences
    #     div_term = torch.exp(torch.arange(0, self.model_dim, 2, device=device).float() * -(math.log(10000.0) / self.model_dim))
        
    #     # Scaling the position indices
    #     pe = torch.zeros(seq_len, self.model_dim, device=device)  # Tensor later filled with positional encodings
    #     pe[:, 0::2] = torch.sin(position * div_term)  # sin applied to the even indices
    #     pe[:, 1::2] = torch.cos(position * div_term)  # cos applied to the odd indices
    #     pe = pe.unsqueeze(0)
    
    #     return pe
        
    # def forward(self, x):
    #     x = x + self._compute_pe(x)
    #     return x
    

class FeedForward(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, ff_dim: int, dropout: float=0.2, swiglu: bool=False):
        """
        FeedForward network with optional SwiGLU activation.
        
        Parameters
        ----------
        input_dim : int
            Input dimension of the feedforward network.
        model_dim : int
            Output dimension of the feedforward network.
        ff_dim : int
            Hidden dimension of the feedforward network.
        dropout : float, optional
            Dropout rate for regularization. Default is 0.2.
        swiglu : bool, optional
            If True, use SwiGLU activation; otherwise, use GELU. Default is False.
        """
        super(FeedForward, self).__init__()
        self.swiglu = swiglu

        if swiglu:
            self.w1 = nn.Linear(input_dim, ff_dim, bias=False)
            self.w2 = nn.Linear(model_dim, ff_dim, bias=False)
            self.w3 = nn.Linear(ff_dim, model_dim, bias=False)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        else:
            self.ff = nn.Sequential(
                nn.Linear(input_dim, ff_dim),
                nn.GELU(approximate='tanh'),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(ff_dim, model_dim),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )

    def forward(self, x):
        if self.swiglu:
            x = F.silu(self.w1(x)) * self.w2(x)
            feed_fwrd = self.w3(self.dropout(x))
            feed_fwrd = self.dropout(feed_fwrd)
        else:
            feed_fwrd = self.ff(x)

        return feed_fwrd
     

import torch
import torch.nn as nn
from transformers import AutoTokenizer

class TextTokenizer(nn.Module):
    def __init__(self, lm_name: str = 't5-small'):
        """
        Tokenizer for the language model.

        Parameters
        ----------
        lm_name : str
            Name of the pre-trained language model to use for tokenization.
            Default is 't5-small'.
        """
        super(TextTokenizer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=True)
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, text: list[str]):
        """
        Tokenize a list of text strings.

        Parameters
        ----------
        text : list[str]
            List of text strings to tokenize.

        Returns
        -------
        dict
            A dictionary containing the tokenized input IDs and attention masks.
        """
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    
    def batch_decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True):
        """
        Decode a batch of token IDs back into text strings.

        Parameters
        ----------
        token_ids : torch.Tensor
            Tensor containing the token IDs to decode.
        skip_special_tokens : bool, optional
            Whether to skip special tokens during decoding. Default is True.

        Returns
        -------
        list[str]
            A list of decoded text strings.
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    

import torch.nn.functional as F

# class ResidualBlock(nn.Module):
#     def __init__(self, 
#                  in_channels: int=1, 
#                  out_channels: int=16, 
#                  kernel_size: int=3, 
#                  dilation: int=2):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.dilation = dilation
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
#         self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

#     def forward(self, x):
#         # Compute left padding so conv remains causal (no lookahead)
#         pad = (self.kernel_size - 1) * self.dilation

#         # First causal convolution (pad only on left)
#         out = F.pad(x, (pad, 0))
#         out = F.relu(self.conv1(out))

#         # Second causal convolution
#         out = F.pad(out, (pad, 0))
#         out = self.conv2(out)

#         # Add residual (identity) connection
#         res = x if self.downsample is None else self.downsample(x)
#         return F.relu(out + res)

# class STCN(nn.Module):
#     def __init__(self, 
#                  num_inputs: int=1, 
#                  num_channels: list=[16, 16], 
#                  output_dim: int=512, 
#                  kernel_size: int=2):
#         """
#         Spatial-Temporal Convolutional Network for human motion sequence embedding.
#         It contains a stack of residual blocks with exponentially increasing dilations,
#         followed by a linear layer to produce the final output.

#         Parameters
#         ----------
#         num_inputs: int
#             Number of input channels (e.g., number of joints in motion data).
#         num_channels: list[int]
#             list of output channels for each TCN layer/block.
#         output_dim: int
#             Dimensionality of the output features.
#         kernel_size: int
#             Filter width of each 1D conv.
#         """
#         super().__init__()
#         layers = []
#         in_ch = num_inputs
#         # Create a stack of ResidualBlocks with dilations 1, 2, 4...
#         for i, ch in enumerate(num_channels):
#             dilation = 2 ** i      # exponentially increasing dilations
#             layers.append(ResidualBlock(in_ch, ch, kernel_size, dilation))
#             in_ch = ch
#         self.tcn = nn.Sequential(*layers)

#         # Final linear layer to produce a single output value
#         self.linear = nn.Linear(num_channels[-1], output_dim)

#     def forward(self, x):
#         """
#         x has shape (batch_size, seq_len, n_joints, 3).
#         """
#         out = self.tcn(x)            # (batch, channels, seq_len)
#         out = out[:, :, -1]          # take the last time step's features
#         out = self.linear(out)       # (batch, output_dim)
#         return out.squeeze(-1)       # return shape (batch, output_dim) 