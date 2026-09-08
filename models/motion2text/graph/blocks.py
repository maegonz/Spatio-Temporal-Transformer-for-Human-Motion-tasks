import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union

class SpatioTemporalAttention(nn.Module):
    """
    Implementation of Spatial-Temporal Multi-Head Self-Attention sub-layer for spatio-temporal graphs,
    based on S2TNet paper https://arxiv.org/pdf/2206.10902

    This layer captures spatial interactions between different nodes (e.g., joints)
    independently at each frame in a temporal sequence. 
    It computes scaled dot-product attention across the spatial and the temporal dimension.

    Parameters
    ----------
    model_dim : int, default=256
        The total dimensionality of the input and output features (D).
    num_heads : int, default=4
        Number of attention heads (H). `model_dim` must be divisible 
        by `num_heads`.
    dropout : float, default=0.2
        Dropout probability applied to the attention scores during the 
        softmax operation.
    mode : str, default="spatial"
        Specifies the mode of attention. Currently supports "spatial" for
        spatial attention across joints and "temporal" for temporal attention.

    Attributes
    ----------
    model_dim : int
        Dimensionality of the model's hidden states.
    num_heads : int
        Number of parallel attention heads.
    head_dim : int
        Dimensionality of each individual attention head (D_head = model_dim // num_heads).
    dropout : torch.nn.Dropout
        Dropout layer for regularization.
    qkv_projection : torch.nn.Linear
        Linear layer to project the input into Query, Key, and Value spaces simultaneously.
    output : torch.nn.Linear
        Final linear projection layer applied after concatenating all attention heads.

    Raises
    ------
    AssertionError
        If `model_dim` is not divisible by `num_heads`.

    Notes
    -----
    This layer expects the Query, Key, and Value tensors to be identical for 
    Self-Attention processing. The spatial attention maps are computed 
    per time step, meaning interactions are localized within each intra-frame.

    Examples
    --------
    >>> batch_size, seq_len, num_joints, model_dim = 2, 6, 22, 256
    >>> spatial_attn = SpatioTemporalAttention(model_dim=model_dim, num_heads=4)
    >>> x = torch.randn(batch_size, seq_len, num_joints, model_dim)
    >>> # For self-attention, Q, K, and V must be the same tensor
    >>> out = spatial_attn(x, x, x)
    >>> out.shape
    torch.Size([2, 6, 22, 256])
    """
    def __init__(self, model_dim: int = 256, num_heads: int = 4, dropout: float = 0.2, mode: str = "spatial"):
        super(SpatioTemporalAttention, self).__init__()
        assert model_dim % num_heads == 0, "Model's dimension must be divisible by num_heads"
        assert mode in ["spatial", "temporal"], "Mode must be either 'spatial' or 'temporal'"

        self.mode = mode
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout_p = dropout

        self.qkv_projection = nn.Linear(model_dim, model_dim * 3)
        self.output = nn.Linear(model_dim, model_dim)

    def forward(self, Q, K, V, mask=None):
        attn_mask = None
        dropout_p = self.dropout_p if self.training else 0.0
        
        if Q is K and K is V:
            # For Self-Attention, Q, K, and V should be the same
            # assert Q.shape == K.shape == V.shape, "Q, K, and V must have the same shape"
            batch_size, seq_len, num_joints, _ = Q.size()

            # [B, S, J, D] -> [B, S, J, 3 * D]
            qkv = self.qkv_projection(Q)
            # [B, S, J, 3, H, D_head]
            qkv = qkv.view(batch_size, seq_len, num_joints, 3, self.num_heads, self.head_dim)

            if self.mode == "spatial":
                # [3, B*S, H, J, D_head]
                qkv = qkv.permute(3, 0, 1, 4, 2, 5).contiguous().view(
                    3, batch_size * seq_len, self.num_heads, num_joints, self.head_dim
                )
            else:
                # [3, B*J, H, S, D_head]
                qkv = qkv.permute(3, 0, 2, 4, 1, 5).contiguous().view(
                    3, batch_size * num_joints, self.num_heads, seq_len, self.head_dim
                )
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Boolean mask necessary (True for keeping, False for masking) or a float mask.
            if mask is not None:
                if mask.dtype != torch.bool:
                    mask = mask.to(torch.bool)
                
                if self.mode == "spatial":
                    # [B, S] -> [B * S, 1, J, J]
                    attn_mask = mask.view(batch_size * seq_len, 1, 1, 1)
                    attn_mask = attn_mask.expand(-1, 1, num_joints, num_joints).contiguous()
                else:
                    if mask.dim() == 3:
                        # [B, S, S] -> [B * J, 1, S, S]
                        attn_mask = mask.unsqueeze(1) # [B, 1, S, S]
                        attn_mask = attn_mask.repeat_interleave(num_joints, dim=0) # [B * J, 1, S, S]
                    else:
                        # [B, S] -> [B * J, 1, S, S]
                        attn_mask = mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, S]
                        attn_mask = attn_mask.repeat_interleave(num_joints, dim=0) # [B * J, 1, 1, S]
                        attn_mask = attn_mask.expand(-1, 1, seq_len, seq_len).contiguous()

            atten_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask, 
                dropout_p=dropout_p
            )

            # [B, S, J, D]
            if self.mode == "spatial":
                atten_output = atten_output.view(batch_size, seq_len, self.num_heads, num_joints, self.head_dim)
                atten_output = atten_output.permute(0, 1, 3, 2, 4).contiguous()
            else:
                atten_output = atten_output.view(batch_size, num_joints, self.num_heads, seq_len, self.head_dim)
                atten_output = atten_output.permute(0, 3, 1, 2, 4).contiguous()

            atten_output = atten_output.view(batch_size, seq_len, num_joints, self.model_dim)

            return self.output(atten_output)
        
        else:
            # For Cross-Attention, Q, K, and V can be different
            # assert K is V, "For Cross-Attention, K and V must be the same"
            batch_size, text_seq_len, _, _ = Q.size()
            _, motion_seq_len, num_joints, _ = K.size()

            # Project Q, K, V separately
            q = F.linear(Q, self.qkv_projection.weight[:self.model_dim], self.qkv_projection.bias[:self.model_dim])
            k = F.linear(K, self.qkv_projection.weight[self.model_dim:2*self.model_dim], self.qkv_projection.bias[self.model_dim:2*self.model_dim])
            v = F.linear(V, self.qkv_projection.weight[2*self.model_dim:], self.qkv_projection.bias[2*self.model_dim:])

            # [B, L, 1, D] -> [B, L, H, D_head]
            q = q.view(batch_size, text_seq_len, self.num_heads, self.head_dim)
            # [B, S, J, D] -> [B, S, J, H, D_head]
            k = k.view(batch_size, motion_seq_len, num_joints, self.num_heads, self.head_dim)
            v = v.view(batch_size, motion_seq_len, num_joints, self.num_heads, self.head_dim)

            if self.mode == "spatial":
                return NotImplementedError("Cross-Attention in spatial mode is not implemented yet.")
                # # [B, L, H, D_head] -> [B*L, H, 1, D_head]
                # q = q.view(batch_size * text_seq_len, self.num_heads, 1, self.head_dim)
                # # [B, S, J, H, D_head] -> [B*S, H, J, D_head]
                # k = k.view(batch_size * motion_seq_len, self.num_heads, num_joints, self.head_dim)
                # v = v.view(batch_size * motion_seq_len, self.num_heads, num_joints, self.head_dim)

            else:
                # [B, L, H, D_head] -> [B, 1, H, L, D_head]
                q = q.permute(0, 2, 1, 3).contiguous().unsqueeze(1) 
                # [B, S, J, H, D_head] -> [B, J, H, S, D_head]
                k = k.permute(0, 2, 3, 1, 4).contiguous()
                v = v.permute(0, 2, 3, 1, 4).contiguous()
            
            # Boolean mask necessary (True for keeping, False for masking) or a float mask.
            if mask is not None:
                if mask.dtype != torch.bool:
                    mask = mask.to(torch.bool)
                
                if self.mode == "spatial":
                    return NotImplementedError("Cross-Attention in spatial mode is not implemented yet.")
                    # # [B, S] -> [B * S, 1, J, J]
                    # attn_mask = mask.view(batch_size * seq_len, 1, 1, 1)
                    # attn_mask = attn_mask.expand(-1, 1, num_joints, num_joints).contiguous()
                else:
                    # [B, S] -> [B, 1, H, L, S]
                    attn_mask = mask[:, None, None, None, :]      # (B, 1, 1, 1, S)
                    attn_mask = attn_mask.expand(-1, 1, self.num_heads, text_seq_len, -1)

            atten_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask, 
                dropout_p=dropout_p
            )

            atten_output = atten_output.view(batch_size, text_seq_len, num_joints, self.num_heads, self.head_dim)
            atten_output = atten_output.view(batch_size, text_seq_len, num_joints, self.model_dim)
            atten_output = torch.mean(atten_output, dim=2).unsqueeze(2)  # Average over joints to get [B, L, 1, D]

            return self.output(atten_output)
    

from models.utils import conv_init, bn_init

class TemporalConv(nn.Module):
    """
    Module de convolution temporelle 1D optimisé sur GPU.
    Traite l'axe temporel (S) de manière indépendante pour chaque joint (J).
    
    Format d'entrée / sortie : [B, S, J, D]
    """
    def __init__(self, model_dim: int = 256, kernel_size: int = 3, dropout: float = 0.2):
        super(TemporalConv, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd to ensure symmetric padding"
        
        self.model_dim = model_dim
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(
            in_channels=model_dim, 
            out_channels=model_dim, 
            kernel_size=kernel_size, 
            padding=padding
        )
        self.bnorm = nn.BatchNorm1d(model_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialisation
        conv_init(self.conv)
        bn_init(self.bnorm, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, J, D]
        batch_size, seq_len, num_joints, model_dim = x.shape

        # [B, S, J, D] -> [B * J, D, S]
        x = x.permute(0, 2, 3, 1).reshape(batch_size * num_joints, model_dim, seq_len)

        x = self.conv(x)
        x = self.bnorm(x)
        x = self.dropout(x)

        # [B * J, D, S] -> [B, S, J, D]
        x = x.view(batch_size, num_joints, model_dim, seq_len)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


from ..transformers.blocks import FeedForward, FastMultiHeadAttention
from typing import Optional

class CrossAttention(nn.Module):
    """
    Cross Attention layer for spatio-temporal graphs, allowing the model to attend to different modalities (e.g., motion and text).
    Computes attention from a query (e.g., motion) to a key-value pair (e.g., text), enabling the model to learn inter-modal relationships.
    It is composed of pre-norm, multi-head attention, and a feed-forward network with residual connections.

    Parameters
    ----------
    model_dim : int, default=256
        The total dimensionality of the input and output features (D).
    num_heads : int, default=8
        Number of attention heads (H). `model_dim` must be divisible 
        by `num_heads`.
    dropout : float, default=0.2
        Dropout probability applied to the attention scores during the softmax operation. 
    """
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.2):
        super(CrossAttention, self).__init__()
        assert model_dim % num_heads == 0, "Model's dimension must be divisible by num_heads"

        self.attn = FastMultiHeadAttention(model_dim, num_heads)
        self.ff = FeedForward(model_dim, model_dim, model_dim * 4, dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_normkv = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, 
                q: torch.Tensor,
                kv: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        # Pre-LN
        q_norm = self.layer_norm1(q)
        kv_norm = self.layer_normkv(kv)

        # Cross Attention
        attn_out = self.attn(q_norm, kv_norm, kv_norm, mask)
        q = q + self.dropout(attn_out)

        # Pre-LN and Feed Forward
        q = self.layer_norm2(q)
        q = q + self.dropout(self.ff(q))
        return q

    

class JointAggregator(nn.Module):
    """
    Learns to aggregate joints using Cross-Attention layer for spatio-temporal graphs.
    This layer computes attention across different joints (nodes) in a graph,
    allowing the model to capture inter-joint dependencies.

    Parameters
    ----------
    model_dim : int, default=256
        The total dimensionality of the input and output features (D).
    num_queries: int, default=4
        Number of query nodes (joints) in the graph. (K)
    num_heads : int, default=8
        Number of attention heads (H). `model_dim` must be divisible 
        by `num_heads`.
    n_layers : int, default=4
        Number of stacked attention layers.
    dropout : float, default=0.2
        Dropout probability applied to the attention scores during the softmax operation.
    """
    def __init__(self, model_dim: int=512,
                 num_joints: int=22,
                 num_queries: int=4,
                 num_heads: int=4,
                 n_layers: int=4,
                 dropout: float=0.2):
        super(JointAggregator, self).__init__()
        assert model_dim % num_heads == 0, "Model's dimension must be divisible by num_heads"

        self.model_dim = model_dim
        self.num_queries = num_queries
        self.num_heads = num_heads

        # Help distinguish between different joints in the graph, for example, the left wrist from the right ankle.
        self.joint_embed = nn.Embedding(100, model_dim)

        self.register_buffer("joint_ids", torch.arange(num_joints), persistent=False) 

        self.queries = nn.Parameter(
            torch.randn(num_queries, model_dim) * (model_dim ** -0.5)
        )
        
        self.layers = nn.ModuleList(
            [CrossAttention(model_dim, num_heads, dropout) for _ in range(n_layers)]
        )
        
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the Joint Aggregator layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, S, J, D], where B is batch size,
            S is sequence length, J is number of joints, and D is model dimension.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, S * K, D] after applying joint aggregation.
        """
        batch_size, seq_len, num_joints, model_dim = x.size()
        total_batch = batch_size * seq_len

        joint_pos = self.joint_embed(self.joint_ids)   # [J, D]
        x = x + joint_pos.view(1, 1, num_joints, model_dim)   # [1, 1, J, D] -> broadcast to [B, S, J, D]
        
        # Reshape input for attention -> [B * S, J, D]
        joints = x.view(total_batch, num_joints, model_dim)
        
        # Expand queries to match batch size and sequence length -> [B * S, K, D]
        queries = self.queries.unsqueeze(0).expand(total_batch, -1, -1)

        attn_mask = None
        if mask is not None:
            attn_mask = mask.view(total_batch, 1, 1, 1)
            attn_mask = attn_mask.expand(-1, 1, self.num_queries, num_joints).contiguous()

        for layer in self.layers:
            queries = layer(queries, joints, mask=attn_mask)
        queries = self.final_norm(queries)  # [B * S, K, D]
        
        # Reshape queries back to [B, S, K, D]
        queries = queries.view(batch_size, seq_len, self.num_queries, model_dim)
        aggregated = queries.flatten(1, 2).contiguous()  # [B, S * K, D]

        mask_out = None
        if mask is not None:
            mask_out = mask.unsqueeze(-1).expand(-1, -1, self.num_queries).reshape(batch_size, -1).contiguous()  # [B, S * K]

        return aggregated, mask_out