import torch
import torch.nn as nn
import math
from .transformers.blocks import PositionalEmbedding, FeedForward, FastMultiHeadAttention
from .transformers.encoders import Encoder
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
                 dropout: float = 0.2,
                 projection_head: bool = True,
                 text_layers: int = 2,
                 swiglu: bool = False):
        """
        Spatial-temporal transformer for motion-to-text generation.

        STformer encodes joint trajectories with spatial-temporal graph and
        temporal encoder layers, decodes target text tokens autoregressively,
        and provides optional projected embeddings for motion-text contrastive
        learning.

        Parameters
        ----------
        motion_dim : int
            Dimensionality of each input motion joint feature.
        tgt_vocab_size : int
            Number of tokens in the target vocabulary.
        model_dim : int, optional
            Dimensionality of model embeddings. Defaults to 512.
        ff_dim : int, optional
            Dimensionality of feed-forward network hidden layers. Defaults to 512.
        num_heads : int, optional
            Number of attention heads. Defaults to 8.
        num_layers : int, optional
            Number of spatial-temporal, temporal, and decoder layers.
            Defaults to 6.
        kernel_size : int, optional
            Kernel size of the spatial-temporal encoders. Defaults to 3.
        dropout : float, optional
            Dropout probability used throughout the model. Defaults to 0.2.
        projection_head : bool, optional
            Whether to project motion and text embeddings for contrastive learning.
            Defaults to True.
        text_layers : int, optional
            Number of text encoder layers. Defaults to 2.
        swiglu : bool, optional
            Whether to use SwiGLU in feed-forward and projection layers.
            Defaults to False.

        Attributes
        ----------
        enc_embedding : FeedForward
            Projects motion features into the model embedding space.
        deco_embedding : torch.nn.Embedding
            Maps target token IDs to model embeddings.
        pos_embedding : PositionalEmbedding
            Adds positional information to motion and text embeddings.
        encoder : torch.nn.ModuleList
            Spatial-temporal and temporal motion encoder layers.
        decoder : torch.nn.ModuleList
            Autoregressive target text decoder layers.
        text_encoder : torch.nn.ModuleList
            Transformer encoder layers used to represent text embeddings.
        motion_proj : torch.nn.Linear, optional
            Projects pooled motion embeddings when projection heads are enabled.
        text_proj : torch.nn.Linear, optional
            Projects pooled text embeddings when projection heads are enabled.
        logit_scale : torch.nn.Parameter
            Learnable log-temperature for contrastive similarity logits.
        projection : FeedForward
            Maps decoder output vectors to target vocabulary logits.
        dropout : torch.nn.Dropout
            Dropout applied to embedded motion and text features.
        """
        super(STformer, self).__init__()
        # Embedding layers
        self.enc_embedding = FeedForward(motion_dim, model_dim, model_dim)
        self.deco_embedding = nn.Embedding(tgt_vocab_size, model_dim)
        self.pos_embedding = PositionalEmbedding(model_dim)
        
        # Motion Encoder layers
        self.encoder = nn.ModuleList(
            [STEncoder(model_dim, num_heads, kernel_size, dropout) for _ in range(num_layers)] + 
            [TEncoder(model_dim, num_heads, dropout, ff_dim) for _ in range(num_layers)]
        )

        # Decoder layers
        self.decoder = nn.ModuleList(
            [TDecoder(model_dim, num_heads, dropout, ff_dim, swiglu=swiglu) for _ in range(num_layers)]
        )

        # self.aggregator = JointAggregator(model_dim=model_dim, num_queries=4, num_heads=4, n_layers=3, dropout=dropout)
        
        # Contrastive layers
        # Text Encoder
        # self.text_encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout, batch_first=True, norm_first=True, num_layers=text_layers, enable_nested_tensor=False)
        # )
        self.text_encoder = nn.ModuleList(
            [Encoder(model_dim, num_heads, dropout, ff_dim, swiglu=swiglu) for _ in range(text_layers)]
        )
        self.projection_head = projection_head
        if projection_head:
            self.motion_proj = nn.Linear(model_dim, model_dim // 2)
            self.text_proj = nn.Linear(model_dim, model_dim // 2)
        # Learnable log-temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))
        
        # Projection layer to map decoder output to target vocabulary size
        self.projection = FeedForward(model_dim, tgt_vocab_size, ff_dim, swiglu=swiglu)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, 
                      tgt: Optional[torch.Tensor]=None,
                      pad_token_id: int=0):
        """
        Generate source and target masks for encoder-decoder architecture.
        Handle optional target inputs during inference or validation process,
        causal masking for autoregressiveness and padding mask.

        Params
        -------
        tgt: torch.Tensor
            input to the decoder
        pad_token_id: int
            ID of the padding token. Defaults to 0.

        Returns
        -------
        returns source mask, target mask, and target padding mask
        """

        tgt_mask, tgt_pad_mask = None, None
        if tgt is not None:
            B, seq_len = tgt.shape
            # Target mask
            tgt_pad_mask = (tgt != pad_token_id) # (batch_size, tgt_seq_len)
            tgt_pad_mask[:, 0] = True
            causal = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device, dtype=torch.bool))  # (tgt_seq_len, tgt_seq_len)
            tgt_mask = causal.unsqueeze(0).expand(B, -1, -1)  # (B,  tgt_seq_len, tgt_seq_len)

        return tgt_mask, tgt_pad_mask if tgt_pad_mask is not None else None

    def forward(self,
                src: torch.Tensor, 
                tgt: Optional[torch.Tensor]=None, 
                encoder_attn_mask: Optional[torch.Tensor]=None,
                max_len: Optional[int] = 64,
                eos_token_id: Optional[int] = 1,
                pad_token_id: Optional[int] = 0,
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
        assert encoder_attn_mask is not None, "pass the collate attn_mask (B, T) bool"
        src_mask = encoder_attn_mask

        # Embedding and positional encoding
        # src: (B, T, 22, 3)
        B, T, J, _ = src.shape

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
        motion_pooled = global_mean_pooling(motion_embeddings, mask=mask_embedddings)  # (B, model_dim)
        motion_embeddings = self.motion_proj(motion_embeddings) if self.projection_head else motion_pooled

        # Apply joint aggregation to get a single representation per frame
        # encoder_output, src_mask = self.aggregator(encoder_output, mask=src_mask)

        if tgt is not None:
            # Training path
            tgt_mask, tgt_pad_mask = self.generate_mask(tgt)

            tgt_emb = self.deco_embedding(tgt)
            tgt_emb = self.pos_embedding(tgt_emb)
            tgt_emb = self.dropout(tgt_emb)

            # Decoder
            decoder_output = tgt_emb
            for decoder in self.decoder:
                decoder_output = decoder(decoder_output, encoder_output, src_mask, tgt_mask)

            # Final linear layer
            logits = self.projection(decoder_output)
            L = logits.size(1)
            assert logits.size(2) == 1, f"Expected singleton dim for logits after projection, got {logits.shape} instead"
            logits = logits.view(B, L, -1)  # [B, T, vocab_size]

            # --- Masked Mean for Text Embeddings ---
            assert tgt_emb.size(2) == 1, f"Expected singleton dim for decoder output before global mean pooling, got {tgt_emb.shape} instead"
            text_encoder = tgt_emb.view(B, L, -1)  # (B, T, model_dim)
            for encoder in self.text_encoder:
                text_encoder = encoder(text_encoder, tgt_mask)
            text_pooled = global_mean_pooling(text_encoder, mask=tgt_pad_mask)  # (B, model_dim)
            text_embeddings = self.text_proj(text_pooled) if self.projection_head else text_pooled

        else:
            # Inference path
            logits = self.generate_description(encoder_output, src_mask, max_len=max_len, eos_token_id=eos_token_id, pad_token_id=pad_token_id, temperature=temperature)
            text_embeddings = None  # No text embeddings during inference

        return {
            "decoder_output": logits,
            "motion_embeddings": motion_embeddings,
            "text_embeddings": text_embeddings,
            "logit_scale": self.logit_scale,
            "loss": None
        }

    @torch.no_grad()  
    def generate_description(self, 
                 encoder_output: torch.Tensor, 
                 src_mask: torch.Tensor,
                 max_len: int = 64,
                 eos_token_id: int = 1,
                 pad_token_id: int = 0,
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
        pad_token_id: int
            Pad token ID. Defaults to 0.
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
        batch_size = encoder_output.size(0)

        # Initialize the decoder input with a start token (assuming start token ID is 0)
        # generated_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=encoder_output.device)  # Start with zeros
        generated_ids = torch.full((batch_size, 1), fill_value=0, dtype=torch.long, device=encoder_output.device)  # Start with start token ID 0
        done = torch.zeros(batch_size, dtype=torch.bool, device=encoder_output.device)

        for _ in range(max_len):  # Maximum generation length
            tgt_emb = self.deco_embedding(generated_ids)
            tgt_emb = self.pos_embedding(tgt_emb)
            tgt_emb = self.dropout(tgt_emb)

            B, L = generated_ids.shape
            causal = torch.tril(torch.ones((L,L), device=generated_ids.device, dtype=torch.bool))
            causal = causal.unsqueeze(0).expand(B, -1, -1)
        
            decoder_output = tgt_emb
            for decoder in self.decoder:
                decoder_output = decoder(decoder_output, encoder_output, src_mask, causal)

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

            next_token_id = next_token_id.masked_fill(done.unsqueeze(-1), pad_token_id)
            # Append the predicted token to the generated sequence
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            # Stop if all sequences have generated an end-of-sequence token (assuming EOS token ID is 1)
            done |= next_token_id.squeeze(-1) == eos_token_id
            if done.all():
                break

        return generated_ids