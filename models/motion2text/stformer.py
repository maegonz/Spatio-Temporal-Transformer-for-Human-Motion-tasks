import torch
import torch.nn as nn
from .transformers.blocks import PositionalEmbedding, FeedForward, FastMultiHeadAttention
from .transformers.decoders import Decoder
from .graph.encoders import STEncoder, TEncoder
from .graph.decoders import TDecoder
from .graph.blocks import JointAggregator
from ..utils import global_mean_pooling
from typing import Optional

class STformer(nn.Module):
    def __init__(self,
                 motion_dim: int,
                 tgt_vocab_size: int,
                 model_dim: int = 512, 
                 ff_dim: int = 512,
                 num_heads: int = 8, 
                 num_layers: int = 6,
                 kernel_size: int = 3, 
                 max_seq_len: int = 100,
                 dropout: float = 0.2,
                 swiglu: bool = False):
        """
        Params
        -------
        motion_dim: int
            Dimensionality of the input motion data.
        tgt_vocab_size: int
            Target vocabulary size.
        model_dim: int
            The dimensionality of the model's embeddings.
        ff_dim: int
            Dimensionality of the inner layer in the feed-forward network.
        num_heads: int
            Number of attention heads in the multi-head attention mechanism.
        num_layers: int
            Number of layers for both the encoder and the decoder.
        kernel_size: int
            Kernel size for the spatial-temporal encoder.
        max_seq_len: int
            Maximum sequence length for positional encoding.
        dropout: float
            Dropout rate for regularization. Defaults to 0.2.
        swiglu: bool
            Whether to use SwiGLU activation function. Defaults to False.
        """
        super(STformer, self).__init__()
        # Embedding layers
        self.enc_embedding = FeedForward(motion_dim, model_dim, model_dim)
        self.deco_embedding = nn.Embedding(tgt_vocab_size, model_dim)
        self.pos_embedding = PositionalEmbedding(model_dim)
        
        # Encoder layers
        self.encoder = nn.ModuleList(
            [STEncoder(model_dim, num_heads, kernel_size, dropout) for _ in range(num_layers)] + 
            [TEncoder(model_dim, num_heads, dropout, ff_dim) for _ in range(num_layers)]
        )

        # Decoder layers
        self.decoder = nn.ModuleList(
            [TDecoder(model_dim, num_heads, dropout, ff_dim, swiglu=swiglu) for _ in range(num_layers)]
        )

        # self.aggregator = JointAggregator(model_dim=model_dim, num_queries=4, num_heads=4, n_layers=3, dropout=dropout)
        
        # Projection layer to map decoder output to target vocabulary size
        self.projection = FeedForward(model_dim, tgt_vocab_size, tgt_vocab_size, swiglu=swiglu)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, 
                      src: torch.Tensor, 
                      tgt: Optional[torch.Tensor]=None,
                      pad_token_id: int=0):
        """
        Generate source and target masks for encoder-decoder architecture.
        Handle optional target inputs during inference or validation process,
        causal masking for autoregressiveness and padding mask.

        Params
        -------
        src: torch.Tensor
            input to the encoder
        tgt: torch.Tensor
            input to the decoder
        pad_token_id: int
            ID of the padding token. Defaults to 0.

        Returns
        -------
        returns source and target masks
        """
        # Source mask
        src_mask = (src != pad_token_id)  # (batch_size, src_seq_len)

        tgt_mask = None
        if tgt is not None:
            seq_len = tgt.size(1)
            # Target mask
            tgt_pad_mask = (tgt != pad_token_id)  # (batch_size, tgt_seq_len)
            tgt_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device, dtype=torch.bool))  # (tgt_seq_len, tgt_seq_len)
            tgt_mask = tgt_pad_mask & tgt_sub_mask  # (batch_size, 1, tgt_seq_len, tgt_seq_len)
            # tgt_pad_mask = (tgt != 0)  # (batch_size, tgt_seq_len)

        return src_mask, tgt_mask


    def forward(self,
                src: torch.Tensor, 
                tgt: Optional[torch.Tensor]=None, 
                encoder_attn_mask: Optional[torch.Tensor]=None,
                max_len: Optional[int] = 50,
                eos_token_id: Optional[int] = 1,
                temperature: Optional[float] = 0.0):
        """
        Run encoder-decoder to predict description sequence.

        Parameters
        ----------
        src : torch.Tensor
           
        tgt : torch.Tensor
            Target description sequence 

        Returns
        -------
        torch.Tensor
            Predicted description sequence of shape (batch, tgt_vocab_size).
        """
        src_mask, _ = self.generate_mask(src, tgt=None)
        src_mask = encoder_attn_mask if encoder_attn_mask is not None else src_mask

        # Embedding and positional encoding
        # src: (B, T, 22, 3)
        B, T, J, _ = src.shape
        device = src.device

        src_emb = self.enc_embedding(src)     # (B, T, J, model_dim)
        src_emb = self.pos_embedding(src_emb)
        src_emb = self.dropout(src_emb)

        # Encoder
        encoder_output = src_emb
        for encoder in self.encoder:
            encoder_output = encoder(encoder_output, src_mask)

        # --- Masked Mean for Motion Embeddings ---
        motion_embeddings = encoder_output.view(B, T * J, -1)  # (B, T*J, model_dim)
        mask_embedddings = src_mask.unsqueeze(-1).expand(-1, -1, J).reshape(B, -1).contiguous()  # (B, T*J)
        motion_embeddings = global_mean_pooling(motion_embeddings, mask=mask_embedddings)  # (B, model_dim)

        # Apply joint aggregation to get a single representation per frame
        # encoder_output, src_mask = self.aggregator(encoder_output, mask=src_mask)

        if tgt is not None:
            # Training path
            _, tgt_mask = self.generate_mask(src, tgt)

            tgt_emb = self.deco_embedding(tgt)
            tgt_emb = self.pos_embedding(tgt_emb)
            tgt_emb = self.dropout(tgt_emb)

            # Decoder
            decoder_output = tgt_emb
            for decoder in self.decoder:
                decoder_output = decoder(decoder_output, encoder_output, src_mask, tgt_mask)

            # Final linear layer
            logits = self.projection(decoder_output)
            logits = logits.squeeze(2)  # [B, T, vocab_size]

            # --- Masked Mean for Text Embeddings ---
            text_embeddings = decoder_output.squeeze(2)  # (B, T, model_dim)
            text_embeddings = global_mean_pooling(text_embeddings, mask=tgt_mask)  # (B, model_dim)

        else:
            # Inference path
            logits = self.generate_description(encoder_output, src_mask, max_len=max_len, eos_token_id=eos_token_id, temperature=temperature)
            text_embeddings = None  # No text embeddings during inference

        return {
            "decoder_output": logits,
            "motion_embeddings": motion_embeddings,
            "text_embeddings": text_embeddings,
            "loss": None
        }

    # @torch.no_grad()
    def generate_description(self, 
                 encoder_output: torch.Tensor, 
                 src_mask: torch.Tensor,
                 max_len: int = 50,
                 eos_token_id: int = 1,
                 temperature: float = 0.0):
        """
        Generate description sequence during inference.

        Parameters
        ----------
        encoder_output : torch.Tensor
            Output from the encoder.
        src_mask : torch.Tensor
            Mask for the source input.
        max_len : int
            Maximum length of the generated sequence.
        eos_token_id : int
            End-of-sequence token ID. Defaults to 1.
        temperature : float
            Temperature for sampling. Defaults to 0.0. 
            If 0, uses greedy decoding, if > 1 flattens
            the probability distribution end up creating
            more diverse outputs, if < 1 sharpens the distribution
            and makes it deterministic, and if = 1 uses 
            the original distribution. 

        Returns
        -------
        torch.Tensor
            Generated description sequence of shape (batch, tgt_vocab_size).
        """
        self.eval()  # Set the model to evaluation mode
        batch_size = encoder_output.size(0)

        # Initialize the decoder input with a start token (assuming start token ID is 0)
        # generated_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=encoder_output.device)  # Start with zeros
        generated_ids = torch.full((batch_size, 1), fill_value=0, dtype=torch.long, device=encoder_output.device)  # Start with start token ID 0

        for _ in range(max_len):  # Maximum generation length
            tgt_emb = self.deco_embedding(generated_ids)
            tgt_emb = self.pos_embedding(tgt_emb)
            tgt_emb = self.dropout(tgt_emb)

            decoder_output = tgt_emb
            for decoder in self.decoder:
                decoder_output = decoder(decoder_output, encoder_output, src_mask, None)

            # Final layer
            logits = self.projection(decoder_output)
            logits = logits.squeeze(2)  # [B, T, vocab_size]

            # Get the last token's logits and sample the next token
            next_token_logits = logits[:, -1, :]  # [B, vocab_size]
            if temperature == 0.0:
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # [B, 1]
            else:
                proba = torch.softmax(next_token_logits / temperature, dim=-1)  # [B, vocab_size]
                next_token_id = torch.multinomial(proba, num_samples=1)  # [B, 1]

            # Append the predicted token to the generated sequence
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            # Stop if all sequences have generated an end-of-sequence token (assuming EOS token ID is 1)
            if (next_token_id == eos_token_id).all():
                break

        return generated_ids