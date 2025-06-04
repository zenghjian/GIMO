import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Tuple, Union, Dict, Any
import os # Added for path manipulation
import matplotlib.pyplot as plt # Added for visualization

# Actual GIMO components
from .pointnet_plus2 import PointNet2SemSegSSGShape, PointNet, MyFPModule
from .base_cross_model import PerceiveEncoder, PositionwiseFeedForward, PerceiveDecoder

# ----- PerceiverAR Implementation Components -----

class ModuleOutput:
    """Output container for PerceiverAR modules."""
    def __init__(self, last_hidden_state, kv_cache=None):
        self.last_hidden_state = last_hidden_state
        self.kv_cache = kv_cache

# Type alias for KV cache
KVCache = Tuple[torch.Tensor, torch.Tensor]  # (key, value)

def positions(batch_size, seq_len, shift=None, device=None):
    """Generate position indices for sequence."""
    pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    if shift is not None:
        pos = pos - shift
    return pos

class RotaryPositionEmbedding:
    """Rotary position embedding for attention mechanisms."""
    def __init__(self, freqs, right_align=False):
        self.freqs = freqs
        self.right_align = right_align

    def apply_rotary_pos_emb(self, x, freqs):
        """Apply rotary position embedding to input tensor."""
        # x shape: [batch_size, seq_len, num_heads, head_dim]
        # freqs shape: [batch_size, seq_len, rotary_dim / 2]
        
        # Make sure we have enough frequency bands for the head dimension
        head_dim = x.shape[-1]
        assert freqs.shape[-1] * 2 >= head_dim, f"Not enough frequency bands ({freqs.shape[-1]*2}) for head dimension ({head_dim})"
        
        # Use only the needed frequencies
        freqs = freqs[..., :head_dim // 2]
        
        # Reshape for broadcasting
        # [batch_size, seq_len, 1, rotary_dim / 2]
        freqs = freqs.unsqueeze(2)
        
        # Split x into even and odd dimensions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # Compute rotations
        # Cosine component for even dimensions
        cos = torch.cos(freqs)
        # Sine component for odd dimensions
        sin = torch.sin(freqs)
        
        # Apply rotation: x_even * cos - x_odd * sin for even positions
        # and x_odd * cos + x_even * sin for odd positions
        x_even_new = x_even * cos - x_odd * sin
        x_odd_new = x_odd * cos + x_even * sin
        
        # Interleave the results
        x_new = torch.zeros_like(x)
        x_new[..., 0::2] = x_even_new
        x_new[..., 1::2] = x_odd_new
        
        return x_new

    def __call__(self, q, k=None):
        """Apply rotary position embeddings to query and optionally key."""
        q_rot = self.apply_rotary_pos_emb(q, self.freqs)
        
        if k is not None:
            # If separate key tensor is provided
            k_rot = self.apply_rotary_pos_emb(k, self.freqs)
            return q_rot, k_rot
        
        return q_rot

class RotarySupport:
    """Mixin to provide rotary position embedding support."""
    def __init__(self, num_input_channels, max_freq=10.0, rotary_dim=None):
        self.num_input_channels = num_input_channels
        self.max_freq = max_freq
        # Default to using all dimensions if rotary_dim not specified
        self.rotary_dim = rotary_dim if rotary_dim is not None else num_input_channels
        # Ensure rotary_dim is even for the rotary encoding
        if self.rotary_dim % 2 != 0:
            self.rotary_dim = self.rotary_dim - 1

    def _compute_freqs(self, positions):
        """
        Compute frequency bands for rotary position embedding.
        
        Args:
            positions: Tensor of shape [batch_size, seq_len]
            
        Returns:
            Tensor of shape [batch_size, seq_len, rotary_dim/2]
        """
        # Get frequency bands
        freq_bands = torch.linspace(0, self.max_freq, self.rotary_dim // 2, device=positions.device)
        # Reshape for broadcasting
        # [1, 1, rotary_dim/2]
        freq_bands = freq_bands.unsqueeze(0).unsqueeze(0)
        # Scale positions by frequency bands
        # [batch_size, seq_len, rotary_dim/2]
        return positions.unsqueeze(-1) * freq_bands

class MultiHeadAttention(nn.Module):
    """Multi-head attention module for cross and self attention."""
    def __init__(
        self,
        num_heads,
        num_q_input_channels,
        num_kv_input_channels=None,
        causal_attention=False,
        dropout=0.0,
        qkv_bias=True,
        out_bias=True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_q_input_channels = num_q_input_channels
        self.num_kv_input_channels = num_kv_input_channels if num_kv_input_channels is not None else num_q_input_channels
        self.causal_attention = causal_attention
        self.dropout = dropout
        
        # Head dimension (channels per head)
        self.head_dim = num_q_input_channels // num_heads
        assert self.head_dim * num_heads == num_q_input_channels, "q_input_channels must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(num_q_input_channels, num_q_input_channels, bias=qkv_bias)
        self.k_proj = nn.Linear(self.num_kv_input_channels, num_q_input_channels, bias=qkv_bias)
        self.v_proj = nn.Linear(self.num_kv_input_channels, num_q_input_channels, bias=qkv_bias)
        self.o_proj = nn.Linear(num_q_input_channels, num_q_input_channels, bias=out_bias)
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize weights with small values for stability
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1/math.sqrt(2))
    
    def forward(
        self,
        q, k=None, v=None,
        rot_pos_emb_q=None,
        rot_pos_emb_k=None,
        attention_mask=None,
        pad_mask=None,
        kv_cache=None
    ):
        """
        Forward pass for multi-head attention.
        
        Args:
            q: Query tensor [batch_size, seq_len_q, channels]
            k: Key tensor [batch_size, seq_len_k, channels] (optional, defaults to q)
            v: Value tensor [batch_size, seq_len_k, channels] (optional, defaults to k)
            rot_pos_emb_q: Rotary position embedding for queries
            rot_pos_emb_k: Rotary position embedding for keys
            attention_mask: Optional attention mask [batch_size, num_heads, seq_len_q, seq_len_k]
            pad_mask: Optional padding mask [batch_size, seq_len_k]
            kv_cache: Optional key-value cache from previous forward passes
            
        Returns:
            Output tensor [batch_size, seq_len_q, channels]
            Updated key-value cache
        """
        batch_size, seq_len_q, _ = q.shape
        
        # Default k and v to q if not provided
        k = q if k is None else k
        v = k if v is None else v
        
        seq_len_k = k.shape[1]
        
        # Project q, k, v
        q = self.q_proj(q)  # [batch_size, seq_len_q, channels]
        k_proj = self.k_proj(k)  # [batch_size, seq_len_k, channels]
        v_proj = self.v_proj(v)  # [batch_size, seq_len_k, channels]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k_proj = k_proj.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v_proj = v_proj.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        # Apply rotary position embeddings if provided
        if rot_pos_emb_q is not None:
            q = rot_pos_emb_q(q)
        
        if rot_pos_emb_k is not None:
            k_proj = rot_pos_emb_k(k_proj)
        
        # Use KV cache if provided
        if kv_cache is not None:
            # Append current k_proj and v_proj to cache
            k_cache, v_cache = kv_cache
            if k_cache is not None and v_cache is not None:
                # Concatenate with cache
                k_proj = torch.cat([k_cache, k_proj], dim=1)
                v_proj = torch.cat([v_cache, v_proj], dim=1)
            
            # Update total sequence length for keys
            seq_len_k = k_proj.shape[1]
            
            # Prepare updated cache
            k_cache_new, v_cache_new = k_proj, v_proj
        else:
            k_cache_new, v_cache_new = None, None
        
        # Rearrange for attention computation
        # [batch_size, num_heads, seq_len_q, head_dim]
        q = q.permute(0, 2, 1, 3)
        # [batch_size, num_heads, seq_len_k, head_dim]
        k_proj = k_proj.permute(0, 2, 1, 3)
        # [batch_size, num_heads, seq_len_k, head_dim]
        v_proj = v_proj.permute(0, 2, 1, 3)
        
        # Compute attention scores
        # [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_scores = torch.matmul(q, k_proj.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply causal mask if needed
        if self.causal_attention and seq_len_q > 1:
            # Create causal mask
            causal_mask = torch.ones(seq_len_q, seq_len_k, device=q.device).triu_(diagonal=1).bool()
            # Apply large negative value to masked positions
            attn_scores.masked_fill_(causal_mask, -1e9)
        
        # Apply padding mask if provided
        if pad_mask is not None:
            # Expand pad_mask for broadcasting
            # [batch_size, 1, 1, seq_len_k]
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
            # Apply large negative value to padded positions
            attn_scores.masked_fill_(pad_mask, -1e9)
        
        # Apply explicit attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Convert scores to probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        attn_probs = self.dropout_layer(attn_probs)
        
        # Compute weighted values
        # [batch_size, num_heads, seq_len_q, head_dim]
        output = torch.matmul(attn_probs, v_proj)
        
        # Reshape back to original dimensions
        # [batch_size, seq_len_q, num_heads, head_dim]
        output = output.permute(0, 2, 1, 3).contiguous()
        # [batch_size, seq_len_q, channels]
        output = output.view(batch_size, seq_len_q, -1)
        
        # Apply output projection
        output = self.o_proj(output)
        
        # Return output and updated cache
        # Always return a tuple of two, (k_cache_new, v_cache_new) will be (None, None) if kv_cache was None
        return output, (k_cache_new, v_cache_new)

class FeedForward(nn.Module):
    """Feed-forward layer with pre-norm."""
    def __init__(self, dim, widening_factor=4, dropout=0.0, residual_dropout=0.0, activation=F.gelu, bias=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * widening_factor, bias=bias)
        self.fc2 = nn.Linear(dim * widening_factor, dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.residual_dropout = nn.Dropout(residual_dropout) if residual_dropout > 0.0 else nn.Identity()
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.residual_dropout(x)
        return residual + x

class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for PerceiverAR."""
    def __init__(
        self,
        num_heads,
        num_q_input_channels,
        num_kv_input_channels=None,
        max_heads_parallel=None,
        causal_attention=True,
        widening_factor=4,
        dropout=0.0,
        residual_dropout=0.0,
        qkv_bias=True,
        out_bias=True,
        mlp_bias=True
    ):
        super().__init__()
        self.norm = nn.LayerNorm(num_q_input_channels)
        self.max_heads_parallel = max_heads_parallel if max_heads_parallel is not None else num_heads
        
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias
        )
        
        self.mlp = FeedForward(
            dim=num_q_input_channels,
            widening_factor=widening_factor,
            dropout=dropout,
            residual_dropout=residual_dropout,
            bias=mlp_bias
        )
    
    def empty_kv_cache(self, q):
        """Initialize an empty KV cache."""
        batch_size = q.shape[0]
        # Return empty tensors for key and value caches
        return (None, None)
    
    def forward(
        self,
        x,
        x_kv_prefix=None,
        pad_mask=None,
        rot_pos_emb_q=None,
        rot_pos_emb_k=None,
        kv_cache=None
    ):
        """
        Forward pass for cross-attention layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, channels]
            x_kv_prefix: Optional prefix for keys and values [batch_size, prefix_len, channels]
            pad_mask: Optional padding mask [batch_size, seq_len + prefix_len]
            rot_pos_emb_q: Rotary position embedding for queries
            rot_pos_emb_k: Rotary position embedding for keys
            kv_cache: Optional key-value cache from previous forward passes
            
        Returns:
            ModuleOutput with:
                - last_hidden_state: Output tensor [batch_size, seq_len, channels]
                - kv_cache: Updated KV cache
        """
        # Normalize input
        x_norm = self.norm(x)
        
        # Prepare k and v inputs
        if x_kv_prefix is not None and x_kv_prefix.shape[1] > 0:
            k = torch.cat([x_kv_prefix, x_norm], dim=1)
            v = k
        else:
            k = x_norm
            v = k
        
        # Apply attention
        output, new_kv_cache = self.attention(
            q=x_norm,
            k=k,
            v=v,
            rot_pos_emb_q=rot_pos_emb_q,
            rot_pos_emb_k=rot_pos_emb_k,
            pad_mask=pad_mask,
            kv_cache=kv_cache
        )
        
        # Apply residual connection
        x = x + output
        
        # Apply feed-forward network
        x = self.mlp(x)
        
        return ModuleOutput(last_hidden_state=x, kv_cache=new_kv_cache)

class SelfAttentionLayer(nn.Module):
    """Self-attention layer for PerceiverAR."""
    def __init__(
        self,
        num_heads,
        num_channels,
        causal_attention=True,
        widening_factor=4,
        dropout=0.0,
        residual_dropout=0.0,
        qkv_bias=True,
        out_bias=True,
        mlp_bias=True
    ):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias
        )
        
        self.mlp = FeedForward(
            dim=num_channels,
            widening_factor=widening_factor,
            dropout=dropout,
            residual_dropout=residual_dropout,
            bias=mlp_bias
        )
    
    def empty_kv_cache(self, x):
        """Initialize an empty KV cache."""
        batch_size = x.shape[0]
        # Return empty tensors for key and value caches
        return (None, None)
    
    def forward(self, x, rot_pos_emb=None, pad_mask=None, kv_cache=None):
        """
        Forward pass for self-attention layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, channels]
            rot_pos_emb: Rotary position embedding
            pad_mask: Optional padding mask [batch_size, seq_len]
            kv_cache: Optional key-value cache from previous forward passes
            
        Returns:
            ModuleOutput with:
                - last_hidden_state: Output tensor [batch_size, seq_len, channels]
                - kv_cache: Updated KV cache
        """
        # Normalize input
        x_norm = self.norm(x)
        
        # Apply attention
        output, new_kv_cache = self.attention(
            q=x_norm,
            rot_pos_emb_q=rot_pos_emb,
            rot_pos_emb_k=rot_pos_emb,
            pad_mask=pad_mask,
            kv_cache=kv_cache
        )
        
        # Apply residual connection
        x = x + output
        
        # Apply feed-forward network
        x = self.mlp(x)
        
        return ModuleOutput(last_hidden_state=x, kv_cache=new_kv_cache)

class SelfAttentionBlock(nn.Module):
    """Block of self-attention layers with optional rotary position embeddings."""
    def __init__(
        self,
        num_layers,
        num_heads,
        num_channels,
        causal_attention=True,
        widening_factor=4,
        dropout=0.0,
        residual_dropout=0.0,
        num_rotary_layers=1,
        activation_checkpointing=False,
        activation_offloading=False,
        qkv_bias=True,
        out_bias=True,
        mlp_bias=True
    ):
        super().__init__()
        self.num_rotary_layers = min(num_rotary_layers, num_layers)
        
        # Create layers
        self.layers = nn.ModuleList([
            SelfAttentionLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                causal_attention=causal_attention,
                widening_factor=widening_factor,
                dropout=dropout,
                residual_dropout=residual_dropout,
                qkv_bias=qkv_bias,
                out_bias=out_bias,
                mlp_bias=mlp_bias
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x, rot_pos_emb=None, pad_mask=None, kv_cache=None):
        """
        Forward pass for self-attention block.
        
        Args:
            x: Input tensor [batch_size, seq_len, channels]
            rot_pos_emb: Rotary position embedding
            pad_mask: Optional padding mask [batch_size, seq_len]
            kv_cache: Optional list of key-value caches from previous forward passes
            
        Returns:
            ModuleOutput with:
                - last_hidden_state: Output tensor [batch_size, seq_len, channels]
                - kv_cache: List of updated KV caches for each layer
        """
        # Initialize kv caches list if not provided
        if kv_cache is None:
            layer_kv_caches = [None] * len(self.layers)
        else:
            layer_kv_caches = kv_cache
        
        # List to store updated KV caches
        new_kv_caches = []
        
        # Process through each layer
        for i, layer in enumerate(self.layers):
            # Apply rotary position embeddings only to specified layers
            current_rot_pos_emb = rot_pos_emb if i < self.num_rotary_layers else None
            
            # Get the KV cache for this layer
            layer_kv_cache = layer_kv_caches[i] if i < len(layer_kv_caches) else None
            
            # Apply the layer
            output = layer(
                x=x,
                rot_pos_emb=current_rot_pos_emb,
                pad_mask=pad_mask,
                kv_cache=layer_kv_cache
            )
            
            # Update the input for the next layer
            x = output.last_hidden_state
            
            # Store the updated KV cache
            if hasattr(output, 'kv_cache') and output.kv_cache is not None:
                new_kv_caches.append(output.kv_cache)
        
        return ModuleOutput(last_hidden_state=x, kv_cache=new_kv_caches)

class TrajectoryInputAdapter(nn.Module, RotarySupport):
    """
    Input adapter for trajectory data to PerceiverAR.
    Transforms trajectory points to the internal representation and provides rotary position embedding support.
    """
    def __init__(self, point_dim, embed_dim, max_freq=10.0):
        nn.Module.__init__(self)
        # RotarySupport.__init__ needs num_input_channels to be embed_dim
        RotarySupport.__init__(self, num_input_channels=embed_dim, max_freq=max_freq, rotary_dim=embed_dim)
        self.point_dim = point_dim # Store point_dim for input check
        
        # Point embedding network
        self.point_embedder = nn.Sequential(
            nn.Linear(point_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
    
    def forward(self, x, abs_pos=None):
        """
        Transform input trajectory points and compute frequency bands for rotary position embeddings.
        
        Args:
            x: Input trajectory points [batch_size, seq_len, point_dim or embed_dim]
            abs_pos: Absolute positions [batch_size, seq_len]
            
        Returns:
            Tuple of:
                - Embedded points [batch_size, seq_len, embed_dim]
                - Frequency bands for rotary embedding [batch_size, seq_len, rotary_dim/2]
        """
        # Embed points if they are raw points, otherwise assume they are already embedded
        if x.shape[-1] == self.point_dim:
            embedded_points = self.point_embedder(x)
        elif x.shape[-1] == self.num_input_channels: # num_input_channels is embed_dim
            embedded_points = x # Already embedded
        else:
            raise ValueError(
                f"Input x last dim {x.shape[-1]} does not match expected point_dim ({self.point_dim}) or embed_dim ({self.num_input_channels})"
            )
        
        # Compute frequency bands for rotary embedding if abs_pos is provided
        if abs_pos is not None:
            freqs = self._compute_freqs(abs_pos)
        else:
            # Create default positions if not provided (e.g., for single token inference without cache)
            batch_size, seq_len = x.shape[0], x.shape[1]
            default_pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            freqs = self._compute_freqs(default_pos)
        
        return embedded_points, freqs

class PerceiverAR(nn.Module):
    """Implementation of Perceiver AR for trajectory forecasting."""
    def __init__(
        self,
        input_adapter,
        num_heads=8,
        max_heads_parallel=None,
        num_self_attention_layers=6,
        num_self_attention_rotary_layers=1,
        self_attention_widening_factor=4,
        cross_attention_widening_factor=4,
        cross_attention_dropout=0.5,
        post_attention_dropout=0.0,
        residual_dropout=0.0,
        activation_checkpointing=False,
        activation_offloading=False
    ):
        """Implementation of Perceiver AR (https://arxiv.org/abs/2202.07765).

    :param input_adapter: Transforms an input sequence to generic Perceiver AR input. An input adapter may choose to
            add (absolute) position information to transformed inputs while `PerceiverAR` additionally computes a
            rotary position embedding (i.e. relative position information) for queries and keys. To support the
            computation of rotary position embeddings, concrete input adapters need to mixin `RotarySupport`.
    :param num_heads: Number of cross- and self-attention heads.
    :param max_heads_parallel: Maximum number of cross-attention heads to be processed in parallel.
        Default is `num_heads`.
    :param num_self_attention_layers: Number of self-attention layers.
    :param cross_attention_dropout: Probability of dropping positions in the prefix sequence.
    :param post_attention_dropout: Probability of dropping cross- and self-attention scores (same as `dropout` in
            Perceiver IO encoder and decoder).
    :param residual_dropout: Probability of dropping residual connections.
    :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention layer and
            cross-attention layer.
    :param activation_offloading: If True, offloads checkpointed activations to CPU.
    """
        super().__init__()
        
        self.input_adapter = input_adapter
        self.cross_attention_dropout = cross_attention_dropout
        
        # Initialize cross-attention layer
        self.cross_attention = CrossAttentionLayer(
            num_heads=num_heads,
            num_q_input_channels=input_adapter.num_input_channels,
            num_kv_input_channels=input_adapter.num_input_channels,
            max_heads_parallel=max_heads_parallel if max_heads_parallel is not None else num_heads,
            causal_attention=True,
            widening_factor=cross_attention_widening_factor,
            dropout=post_attention_dropout,
            residual_dropout=residual_dropout,
            qkv_bias=False,
            out_bias=True,
            mlp_bias=False
        )
        
        # Initialize self-attention blocks
        self.self_attention = SelfAttentionBlock(
            num_layers=num_self_attention_layers,
            num_heads=num_heads,
            num_channels=input_adapter.num_input_channels,
            causal_attention=True,
            widening_factor=self_attention_widening_factor,
            dropout=post_attention_dropout,
            residual_dropout=residual_dropout,
            num_rotary_layers=num_self_attention_rotary_layers,
            activation_checkpointing=activation_checkpointing,
            activation_offloading=activation_offloading,
            qkv_bias=False,
            out_bias=False,
            mlp_bias=False
        )
    
    def forward(
        self,
        x,
        prefix_len,
        pad_mask=None,
        kv_cache=None
    ):
        """
        Forward pass for PerceiverAR.
        
        Args:
            x: Input tensor with context and target sequence concatenated [batch_size, context_len + target_len, channels]
            prefix_len: Length of the context prefix
            pad_mask: Optional padding mask [batch_size, context_len + target_len]
            kv_cache: Optional key-value cache from previous forward passes
            
        Returns:
            ModuleOutput with:
                - last_hidden_state: Output tensor [batch_size, target_len, channels]
                - kv_cache: Updated KV cache
        """
        if pad_mask is None:
            shift = None
        else:
            # Sum padding mask along sequence dimension to get shift for each batch item
            shift = pad_mask.sum(dim=1, keepdim=True)
        
        # Determine sequence length based on input and cache
        if kv_cache is None or len(kv_cache) == 0:
            # No cache or empty cache, use input shape
            b, n = x.shape[0], x.shape[1]
        else:
            # Cache exists and non-empty, consider cache size in total sequence length
            b = x.shape[0]
            n = kv_cache[0][0].shape[1] + x.shape[1] if kv_cache[0][0] is not None else x.shape[1]
        
        # Validate prefix length
        if not 0 <= prefix_len < n:
            raise ValueError(f"prefix_len ({prefix_len}) out of valid range [0..{n})")
        
        # Apply input adapter to get embedded representation and frequency position encodings
        # positions() generates absolute positions for the sequence
        x, frq_pos_enc = self.input_adapter(x, abs_pos=positions(b, n, shift=shift, device=x.device))
        
        # Extract latent (target) and prefix (context) parts based on cache state
        if kv_cache is None or len(kv_cache) == 0:
            # No cache or empty cache, split input into prefix and latent
            x_latent = x[:, prefix_len:]  # Target sequence
            x_prefix = x[:, :prefix_len]  # Context
        else:
            # Cache exists, use entire input as latent and empty prefix
            x_latent = x
            x_prefix = x[:, :0]  # Empty prefix
        
        # Split frequency position encodings accordingly
        frq_pos_enc_latent = frq_pos_enc[:, prefix_len:]
        frq_pos_enc_prefix = frq_pos_enc[:, :prefix_len]
        
        # Process padding mask if provided
        if pad_mask is not None:
            pad_mask_latent = pad_mask[:, prefix_len:]
            pad_mask_prefix = pad_mask[:, :prefix_len]
        
        # Apply dropout to context during training (only when no cache is used)
        if self.training and prefix_len > 0 and self.cross_attention_dropout > 0.0:
            if kv_cache is not None:
                raise ValueError("cross-attention dropout not supported with caching")
            
            # Randomly keep a subset of context positions
            rand = torch.rand(b, prefix_len, device=x.device)
            keep = prefix_len - int(prefix_len * self.cross_attention_dropout)
            keep_indices = rand.topk(keep, dim=-1).indices
            keep_mask = torch.zeros_like(rand, dtype=torch.bool).scatter_(dim=1, index=keep_indices, value=1)
            
            # Apply mask to context
            x_prefix = x_prefix[keep_mask].view(b, -1, x_prefix.size(-1))
            frq_pos_enc_prefix = frq_pos_enc_prefix[keep_mask].view(b, -1, frq_pos_enc_prefix.size(-1))
            
            if pad_mask is not None:
                pad_mask_prefix = pad_mask_prefix[keep_mask].view(b, -1)
        
        # Prepare rotary position embeddings
        frq_pos_enc_q = frq_pos_enc_latent
        frq_pos_enc_k = torch.cat([frq_pos_enc_prefix, frq_pos_enc_latent], dim=1)
        
        # Combine padding masks if needed
        if pad_mask is not None:
            pad_mask = torch.cat([pad_mask_prefix, pad_mask_latent], dim=1)
        
        # Initialize or extract KV caches
        if kv_cache is None:
            ca_kv_cache = None
            sa_kv_cache = None
            kv_cache_updated = None
        elif len(kv_cache) == 0:
            ca_kv_cache = self.cross_attention.empty_kv_cache(x_latent)
            sa_kv_cache = []
            kv_cache_updated = []
        else:
            ca_kv_cache, *sa_kv_cache = kv_cache
            kv_cache_updated = []
        
        # Apply cross-attention between target sequence and context
        ca_output = self.cross_attention(
            x=x_latent,
            x_kv_prefix=x_prefix,
            pad_mask=pad_mask,
            rot_pos_emb_q=RotaryPositionEmbedding(frq_pos_enc_q, right_align=True),
            rot_pos_emb_k=RotaryPositionEmbedding(frq_pos_enc_k, right_align=True),
            kv_cache=ca_kv_cache
        )
        
        # Store updated cross-attention KV cache
        if kv_cache_updated is not None:
            kv_cache_updated.append(ca_output.kv_cache)
        
        # Apply self-attention to cross-attention output
        sa_output = self.self_attention(
            x=ca_output.last_hidden_state,
            rot_pos_emb=RotaryPositionEmbedding(frq_pos_enc_latent, right_align=True),
            kv_cache=sa_kv_cache
        )
        
        # Store updated self-attention KV caches
        if kv_cache_updated is not None:
            kv_cache_updated.extend(sa_output.kv_cache)
        
        # Return final output and updated caches
        return ModuleOutput(last_hidden_state=sa_output.last_hidden_state, kv_cache=kv_cache_updated)

class GIMO_ADT_Autoregressive_Model(nn.Module):
    """
    GIMO Autoregressive model for ADT Object Motion Prediction.
    Uses the PointNet++ for scene encoding and Perceiver architecture for motion encoding.
    Implements Perceiver AR for autoregressive trajectory forecasting.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Store history fraction for dynamic splitting
        self.history_fraction = config.history_fraction
        
        # Check if text embedding is disabled
        self.use_text_embedding = not getattr(config, 'no_text_embedding', False)
        
        # Check if bounding box processing is disabled
        self.use_bbox = not getattr(config, 'no_bbox', False)
        
        # Setup dimensions
        self.sequence_length = config.trajectory_length 
        self.autoregressive_latent_dim = 128
        self.perceiver_ar_dim = 256  # Dimension for Perceiver AR

        # --- Scene/Motion Encoding Components (Perceiver IO) ---

        # 1. Motion Embedding
        self.motion_linear = nn.Linear(config.object_motion_dim, config.motion_hidden_dim)

        # 2. Point Cloud Encoder
        self.scene_encoder = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})

        # Bounding Box related layers
        self.fp_layer = MyFPModule()
        self.bbox_pointnet = PointNet(config.scene_feats_dim)
        
        # Encoder for fusing motion and bbox_scene_features
        if not self.use_bbox:
            # If bbox disabled, we only use motion features
            self.motion_bbox_encoder = PerceiveEncoder(
                n_input_channels=config.motion_hidden_dim,
                n_latent=self.autoregressive_latent_dim,
                n_latent_channels=config.motion_latent_dim,
                n_self_att_heads=config.motion_n_heads,    
                n_self_att_layers=config.motion_n_layers,   
                dropout=config.dropout
            )
        else:
            # With bbox enabled, we use both motion and bbox features
            self.motion_bbox_encoder = PerceiveEncoder(
                n_input_channels=config.motion_hidden_dim + config.scene_feats_dim,
                n_latent=self.autoregressive_latent_dim, 
                n_latent_channels=config.motion_latent_dim,
                n_self_att_heads=config.motion_n_heads,    
                n_self_att_layers=config.motion_n_layers,   
                dropout=config.dropout
            )

        # --- Category Embedding ---
        if not config.no_text_embedding:
            self.category_embedding = nn.Embedding(
                num_embeddings=config.num_object_categories,
                embedding_dim=config.category_embed_dim
            )
        
        # Calculate embedding dimension
        embed_dim = config.motion_latent_dim + config.scene_feats_dim
        if not config.no_text_embedding:
            embed_dim += config.category_embed_dim
        
        # Processing of fused context features
        self.embedding_layer = PositionwiseFeedForward(
            d_in=embed_dim, 
            d_hid=embed_dim, 
            dropout=config.dropout
        )
        
        # Output encoder to process fused context
        self.context_encoder = PerceiveEncoder(
            n_input_channels=embed_dim,
            n_latent=self.autoregressive_latent_dim, 
            n_latent_channels=self.perceiver_ar_dim,  # Adjust to match Perceiver AR dimension
            n_self_att_heads=config.output_n_heads,
            n_self_att_layers=config.output_n_layers,
            dropout=config.dropout
        )
        
        # --- Autoregressive Trajectory Prediction Components ---
        
        # 1. Input adapters for Perceiver AR
        self.trajectory_input_adapter = TrajectoryInputAdapter(
            point_dim=config.object_motion_dim,  # 6D motion (position + orientation)
            embed_dim=self.perceiver_ar_dim,     # Match context dimension
            max_freq=10.0                        # Maximum frequency for rotary embeddings
        )
        
        # 2. Linear projections for mapping between different spaces
        self.context_projection = nn.Linear(
            self.perceiver_ar_dim,  # From context encoder output
            self.perceiver_ar_dim   # To Perceiver AR input dimension
        )
        
        # 3. Perceiver AR for autoregressive decoding
        self.perceiver_ar = PerceiverAR(
            input_adapter=self.trajectory_input_adapter,
            num_heads=config.output_n_heads,      # Use same as context
            num_self_attention_layers=config.output_n_layers,  # Use same as context
            cross_attention_dropout=0.1,          # Dropout for cross-attention
            post_attention_dropout=config.dropout # Use same dropout as context
        )
        
        # 4. Regression head for final trajectory point prediction
        self.regression_head = nn.Sequential(
            nn.Linear(self.perceiver_ar_dim, self.perceiver_ar_dim),
            nn.LayerNorm(self.perceiver_ar_dim),
            nn.GELU(),
            nn.Linear(self.perceiver_ar_dim, config.object_motion_dim)
        )
    
    def encode_context(self, input_trajectory, point_cloud, bounding_box_corners=None, object_category_ids=None):
        """
        Encode context information (scene, history, object category) using Perceiver IO.
        
        Returns:
            Context tensor [B, autoregressive_latent_dim, perceiver_ar_dim]
        """
        batch_size = input_trajectory.shape[0]
        input_length = input_trajectory.shape[1]
        
        # --- Feature Extraction ---
        f_m = self.motion_linear(input_trajectory)
        point_cloud_6d = torch.cat([point_cloud, point_cloud], dim=2)
        scene_feats_per_point, scene_global_feats = self.scene_encoder(point_cloud_6d)
        
        if self.use_bbox and bounding_box_corners is not None:
            num_bbox_corners = bounding_box_corners.shape[2]
            point_cloud_repeated_for_fp = point_cloud.unsqueeze(1).repeat(1, input_length, 1, 1)
            point_cloud_for_fp = point_cloud_repeated_for_fp.reshape(batch_size * input_length, -1, 3).contiguous()
            scene_feats_per_point_for_fp = scene_feats_per_point.unsqueeze(2).repeat(1, 1, input_length, 1)
            scene_feats_per_point_for_fp = scene_feats_per_point_for_fp.permute(0, 2, 1, 3).reshape(batch_size * input_length, self.config.scene_feats_dim, -1).contiguous()
            bbox_corners_for_fp = bounding_box_corners.reshape(batch_size * input_length, num_bbox_corners, 3).contiguous()
            propagated_feats_to_bbox = self.fp_layer(
                unknown=bbox_corners_for_fp, 
                known=point_cloud_for_fp,    
                known_feats=scene_feats_per_point_for_fp
            )
            f_s_b_per_ts = self.bbox_pointnet(propagated_feats_to_bbox) 
            f_s_b = f_s_b_per_ts.reshape(batch_size, input_length, self.config.scene_feats_dim)
            fused_motion_bbox_input = torch.cat([f_m, f_s_b], dim=2)
        else:
            fused_motion_bbox_input = f_m
        
        # encoded_motion_bbox shape: [B, self.autoregressive_latent_dim, motion_latent_dim]
        encoded_motion_bbox = self.motion_bbox_encoder(fused_motion_bbox_input)

        # --- Category Embeddings ---
        category_embeddings_expanded = None
        if not self.config.no_text_embedding and object_category_ids is not None:
            category_embeddings = self.category_embedding(object_category_ids)
            # Repeat for each latent token
            category_embeddings_expanded = category_embeddings.unsqueeze(1).repeat(1, self.autoregressive_latent_dim, 1)

        # --- Fusion ---
        # Expand scene features to match latent dimension
        scene_global_feats_expanded = scene_global_feats.unsqueeze(1).repeat(1, self.autoregressive_latent_dim, 1)
        
        features_to_fuse = [
            scene_global_feats_expanded,
            encoded_motion_bbox,
        ]

        if category_embeddings_expanded is not None:
            features_to_fuse.append(category_embeddings_expanded)
            
        final_fused_input = torch.cat(features_to_fuse, dim=2)
        
        # Process fused features through embedding layer and context encoder
        cross_modal_embedding = self.embedding_layer(final_fused_input) 
        context = self.context_encoder(cross_modal_embedding)
        
        # Project context to match Perceiver AR dimensions
        projected_context = self.context_projection(context)
        
        return projected_context
    
    def forward(self, input_trajectory, point_cloud, bounding_box_corners=None, object_category_ids=None, 
                semantic_bbox_info=None, semantic_bbox_mask=None, semantic_text_categories=None):
        """
        Forward pass with autoregressive trajectory prediction.
        
        Args:
            input_trajectory: [batch_size, input_length, 6] - history trajectory
            point_cloud: [batch_size, num_points, 3] - scene point cloud
            bounding_box_corners: [batch_size, input_length, 8, 3] (optional)
            object_category_ids: [batch_size] (optional)
            semantic_bbox_info: [batch_size, max_bboxes, 12] - semantic bbox information for scene conditioning (optional)
            semantic_bbox_mask: [batch_size, max_bboxes] - mask for semantic bbox (optional)
            semantic_text_categories: List of category strings for semantic text conditioning (optional)
            
        Returns:
            Predicted full trajectory [batch_size, sequence_length, 6]
        """
        batch_size = input_trajectory.shape[0]
        input_length = input_trajectory.shape[1]
        device = input_trajectory.device
        
        # Check for required inputs
        if not self.config.no_text_embedding and object_category_ids is None:
            raise ValueError("object_category_ids is required when no_text_embedding=False")
        if self.use_bbox and bounding_box_corners is None:
            raise ValueError("bounding_box_corners is required when no_bbox=False")
            
        # Encode context using Perceiver IO
        # context shape: [B, self.autoregressive_latent_dim, self.perceiver_ar_dim]
        context = self.encode_context(
            input_trajectory=input_trajectory,
            point_cloud=point_cloud,
            bounding_box_corners=bounding_box_corners,
            object_category_ids=object_category_ids
        )
        
        # Determine target number of future frames to predict
        # If training, this might be fixed or based on gt_future_trajectory length
        # If inferring, this is sequence_length - input_length
        num_future_frames = self.sequence_length - input_length
        if num_future_frames <= 0:
            # If sequence_length is not longer than input_length, return input_trajectory or part of it
            return input_trajectory[:, :self.sequence_length, :]

        # Autoregressive generation loop
        # Start with the last observed point from input_trajectory
        if input_length > 0:
            current_raw_point = input_trajectory[:, -1:, :]  # [B, 1, 6]
        else:
            # If no history (e.g. use_first_frame_only with input_length=0, or predicting from scratch)
            # Create a zero start token. Its position will be 0.
            current_raw_point = torch.zeros((batch_size, 1, self.config.object_motion_dim), device=device)

        all_predicted_raw_points = [] # To store raw [B,1,6] predictions
        kv_cache = None

        for k in range(num_future_frames):
            # Embed the current raw point to perceiver_ar_dim
            # The point_embedder part of TrajectoryInputAdapter handles nn.Linear(6, perceiver_ar_dim)
            current_point_embedding = self.trajectory_input_adapter.point_embedder(current_raw_point) # [B, 1, perceiver_ar_dim]

            if k == 0: # First step of AR generation
                # x_for_ar is concatenation of context and current (first) query point embedding
                x_for_ar = torch.cat([context, current_point_embedding], dim=1) # [B, L_ctx + 1, D_ar_dim]
                prefix_len = context.shape[1] # Length of the context part
            else: # Subsequent steps
                # x_for_ar is just the current query point embedding
                # Context is managed by kv_cache within PerceiverAR
                x_for_ar = current_point_embedding # [B, 1, D_ar_dim]
                prefix_len = 0
           
            # Forward pass through Perceiver AR
            # The internal input_adapter of perceiver_ar (our modified TrajectoryInputAdapter)
            # will see x_for_ar (already in perceiver_ar_dim), pass features, and compute RoPE freqs.
            ar_output = self.perceiver_ar(\
                x=x_for_ar,\
                prefix_len=prefix_len,\
                kv_cache=kv_cache\
            )\

            # Update KV cache for next step
            kv_cache = ar_output.kv_cache
           
            # Get hidden state for the current query point (it's the last in the sequence output by AR)
            # ar_output.last_hidden_state is [B, query_len, D_ar_dim]. Here query_len is 1.
            hidden_state_current_step = ar_output.last_hidden_state[:, -1:, :] # [B, 1, D_ar_dim]
           
            # Predict next raw 6D point
            next_raw_point = self.regression_head(hidden_state_current_step)  # [B, 1, 6]
           
            all_predicted_raw_points.append(next_raw_point)
            current_raw_point = next_raw_point # Use predicted point as input for the next step
       
        # Concatenate all predicted future raw points
        if len(all_predicted_raw_points) > 0:
            predicted_future_trajectory = torch.cat(all_predicted_raw_points, dim=1)  # [B, num_future_frames, 6]
        else: # Should not happen if num_future_frames > 0
            predicted_future_trajectory = torch.empty((batch_size, 0, self.config.object_motion_dim), device=device)

        # Concatenate with input_trajectory to form the full predicted trajectory
        # The model is expected to output a trajectory of self.sequence_length
        if input_length > 0:
            full_predicted_trajectory = torch.cat([input_trajectory, predicted_future_trajectory], dim=1)
        else: # Predicting from scratch, future is the full trajectory
            full_predicted_trajectory = predicted_future_trajectory
        
        # Ensure final trajectory is of `self.sequence_length` (it should be by construction)
        # This might truncate if num_future_frames was miscalculated or if input was longer than sequence_length initially
        return full_predicted_trajectory[:, :self.sequence_length, :]
    
    def compute_loss(self, predictions, batch, epoch=None, batch_idx=None, vis_save_dir=None, sample_name_for_vis=None):
        """
        Computes the loss for the GIMO_ADT_Autoregressive_Model.
        Loss calculation is similar to GIMO_ADT_Model but with a focus on the autoregressive nature.

        Args:
            predictions: The model's output tensor [B, sequence_length, 6]
            batch: The batch dictionary with ground truth data
            epoch (optional): Current epoch number, for visualization.
            batch_idx (optional): Current batch index, for visualization.
            vis_save_dir (optional): Directory to save visualizations.
            sample_name_for_vis (optional): Sample name for visualization filename.

        Returns:
            Total loss and loss components dictionary
        """
        gt_full_poses = batch['full_poses'].to(predictions.device)
        gt_full_mask = batch['full_attention_mask'].to(predictions.device)
        device = predictions.device
        position_dim = 3
        batch_size = gt_full_poses.shape[0] # Added batch_size for visualization

        gt_full_positions = gt_full_poses[..., :position_dim]
        gt_full_orientations = gt_full_poses[..., position_dim:]
        pred_positions = predictions[..., :position_dim]
        pred_orientations = predictions[..., position_dim:]
        
        # Calculate dynamic split lengths 
        actual_lengths = torch.sum(gt_full_mask, dim=1)
        if self.config.use_first_frame_only:
            dynamic_history_lengths = torch.ones_like(actual_lengths).long()
            dynamic_history_lengths = torch.min(dynamic_history_lengths, actual_lengths.long())
        else:
            min_val_tensor = torch.tensor(1, device=device)
            dynamic_history_lengths = torch.floor(actual_lengths * self.history_fraction).long().clamp(min=min_val_tensor, max=actual_lengths.long())
        
        indices = torch.arange(self.sequence_length, device=device).unsqueeze(0)
        dynamic_future_mask = (indices >= dynamic_history_lengths.unsqueeze(1)) * gt_full_mask
        
        if self.config.use_first_frame_only:
            future_only_mask = (indices >= 1).float() * gt_full_mask
            dynamic_future_mask_for_loss = future_only_mask
        else:
            dynamic_future_mask_for_loss = dynamic_future_mask
        
        # Calculate translation loss
        loss_trans_per_coord = F.l1_loss(pred_positions, gt_full_positions, reduction='none')
        loss_trans_per_point = torch.sum(loss_trans_per_coord, dim=-1)
        masked_loss_trans = loss_trans_per_point * dynamic_future_mask_for_loss
        
        sum_loss_trans_per_seq = torch.sum(masked_loss_trans, dim=1)
        num_valid_points_per_seq = torch.sum(dynamic_future_mask_for_loss, dim=1)
        valid_trans_mask = num_valid_points_per_seq > 0
        mean_loss_trans_per_seq = torch.zeros_like(sum_loss_trans_per_seq)
        if torch.any(valid_trans_mask):
            mean_loss_trans_per_seq[valid_trans_mask] = sum_loss_trans_per_seq[valid_trans_mask] / num_valid_points_per_seq[valid_trans_mask]
        mean_trans_loss = torch.sum(mean_loss_trans_per_seq) / torch.sum(valid_trans_mask) if torch.sum(valid_trans_mask) > 0 else torch.tensor(0.0, device=device)
        
        # Calculate orientation loss
        loss_ori_per_coord = F.l1_loss(pred_orientations, gt_full_orientations, reduction='none')
        loss_ori_per_point = torch.sum(loss_ori_per_coord, dim=-1)
        masked_loss_ori = loss_ori_per_point * dynamic_future_mask_for_loss
        
        sum_loss_ori_per_seq = torch.sum(masked_loss_ori, dim=1)
        mean_loss_ori_per_seq = torch.zeros_like(sum_loss_ori_per_seq)
        if torch.any(valid_trans_mask):
            mean_loss_ori_per_seq[valid_trans_mask] = sum_loss_ori_per_seq[valid_trans_mask] / num_valid_points_per_seq[valid_trans_mask]
        mean_ori_loss = torch.sum(mean_loss_ori_per_seq) / torch.sum(valid_trans_mask) if torch.sum(valid_trans_mask) > 0 else torch.tensor(0.0, device=device)
        
        # Calculate reconstruction loss for input trajectory
        mean_rec_loss = torch.tensor(0.0, device=device)
        if not self.config.use_first_frame_only and dynamic_history_lengths.max() > 0:
            hist_mask = (indices < dynamic_history_lengths.unsqueeze(1)) * gt_full_mask
            loss_rec_per_coord = F.l1_loss(predictions, gt_full_poses, reduction='none')
            loss_rec_per_point = torch.sum(loss_rec_per_coord, dim=-1)
            masked_loss_rec = loss_rec_per_point * hist_mask
            
            sum_loss_rec_per_seq = torch.sum(masked_loss_rec, dim=1)
            num_valid_hist_points_per_seq = torch.sum(hist_mask, dim=1)
            valid_hist_mask = num_valid_hist_points_per_seq > 0
            mean_loss_rec_per_seq = torch.zeros_like(sum_loss_rec_per_seq)
            if torch.any(valid_hist_mask):
                mean_loss_rec_per_seq[valid_hist_mask] = sum_loss_rec_per_seq[valid_hist_mask] / num_valid_hist_points_per_seq[valid_hist_mask]
            mean_rec_loss = torch.sum(mean_loss_rec_per_seq) / torch.sum(valid_hist_mask) if torch.sum(valid_hist_mask) > 0 else torch.tensor(0.0, device=device)
        elif self.config.use_first_frame_only:
            first_frame_gt = gt_full_poses[:, 0, :]
            first_frame_pred = predictions[:, 0, :]
            first_frame_mask = gt_full_mask[:, 0]
            first_frame_loss = F.l1_loss(first_frame_pred, first_frame_gt, reduction='none')
            first_frame_loss_per_batch = torch.sum(first_frame_loss, dim=1)
            valid_first_frames = torch.sum(first_frame_mask)
            if valid_first_frames > 0:
                masked_first_frame_loss = first_frame_loss_per_batch * first_frame_mask
                mean_rec_loss = torch.sum(masked_first_frame_loss) / valid_first_frames
        
        # Apply loss weights
        lambda_trans = getattr(self.config, 'lambda_trans', 1.0)
        lambda_ori = getattr(self.config, 'lambda_ori', 1.0)
        lambda_rec = getattr(self.config, 'lambda_rec', 1.0)
        
        weighted_trans_loss = lambda_trans * mean_trans_loss
        weighted_ori_loss = lambda_ori * mean_ori_loss
        weighted_rec_loss = lambda_rec * mean_rec_loss
        
        # Calculate destination loss
        last_valid_indices = (torch.sum(gt_full_mask, dim=1) - 1).long().clamp(min=0)
        batch_indices = torch.arange(gt_full_poses.shape[0], device=device)
        last_gt_poses = gt_full_poses[batch_indices, last_valid_indices]
        last_pred_poses = predictions[batch_indices, last_valid_indices]
        dest_loss_per_coord = F.l1_loss(last_pred_poses, last_gt_poses, reduction='none')
        dest_trans_loss_per_seq = torch.sum(dest_loss_per_coord[:, :position_dim], dim=1)
        dest_ori_loss_per_seq = torch.sum(dest_loss_per_coord[:, position_dim:], dim=1)
        valid_dest_mask = last_valid_indices >= 0
        dest_trans_loss = torch.sum(dest_trans_loss_per_seq * valid_dest_mask.float()) / torch.sum(valid_dest_mask.float()) if torch.sum(valid_dest_mask) > 0 else torch.tensor(0.0, device=device)
        dest_ori_loss = torch.sum(dest_ori_loss_per_seq * valid_dest_mask.float()) / torch.sum(valid_dest_mask.float()) if torch.sum(valid_dest_mask) > 0 else torch.tensor(0.0, device=device)
        
        weighted_dest_trans_loss = lambda_trans * dest_trans_loss 
        weighted_dest_ori_loss = lambda_ori * dest_ori_loss 
        
        # Compute total loss
        total_loss = weighted_trans_loss + weighted_ori_loss + weighted_rec_loss + weighted_dest_trans_loss + weighted_dest_ori_loss
                     
        # Return loss dictionary
        loss_dict = {
            'total_loss': total_loss,
            'trans_loss': weighted_trans_loss,
            'ori_loss': weighted_ori_loss,
            'rec_loss': weighted_rec_loss,
            'dest_trans_loss': weighted_dest_trans_loss,
            'dest_ori_loss': weighted_dest_ori_loss
        }
            
        # --- Visualization Logic (similar to GIMO_ADT_Model) ---
        if vis_save_dir and sample_name_for_vis and epoch is not None and batch_idx is not None:
            try:
                if batch_size > 0:
                    os.makedirs(vis_save_dir, exist_ok=True)

                    # Re-calculate masks needed for visualization if they were not stored or are different
                    # In this model, dynamic_history_lengths and dynamic_future_mask are already computed.
                    # We need dynamic_history_mask specifically for visualization if not used directly above.
                    indices_vis = torch.arange(self.sequence_length, device=device).unsqueeze(0)
                    dynamic_history_mask_vis = (indices_vis < dynamic_history_lengths.unsqueeze(1)).float()
                    # dynamic_future_mask_for_loss is already computed and suitable for visualization.

                    gt_poses_vis = gt_full_poses[0, :, :position_dim].norm(dim=-1).cpu().numpy()
                    gt_mask_vis = gt_full_mask[0].cpu().numpy()
                    hist_mask_plot = dynamic_history_mask_vis[0].cpu().numpy() # Use the specifically created one
                    future_mask_plot = dynamic_future_mask[0].cpu().numpy() # Changed to use dynamic_future_mask directly
                    hist_len_vis = dynamic_history_lengths[0].item()

                    timesteps = np.arange(self.sequence_length)

                    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                    fig.suptitle(f'Mask Visualization - {sample_name_for_vis}\\nEpoch {epoch}, Batch {batch_idx}, Hist Len: {hist_len_vis}', fontsize=10)

                    axs[0].bar(timesteps, gt_poses_vis, color='blue', alpha=0.7)
                    axs[0].set_title('GT Full Pose (Position L2 Norm)', fontsize=8)
                    axs[0].set_ylabel('L2 Norm', fontsize=8)

                    axs[1].bar(timesteps, gt_mask_vis, color='green', alpha=0.7)
                    axs[1].set_title('GT Full Mask', fontsize=8)
                    axs[1].set_ylabel('Mask Value', fontsize=8)
                    axs[1].set_yticks([0, 1])

                    axs[2].bar(timesteps, hist_mask_plot, color='red', alpha=0.7)
                    axs[2].set_title(f'Dynamic History Mask (Length: {hist_len_vis})', fontsize=8)
                    axs[2].set_ylabel('Mask Value', fontsize=8)
                    axs[2].set_yticks([0, 1])

                    axs[3].bar(timesteps, future_mask_plot, color='purple', alpha=0.7)
                    axs[3].set_title('Dynamic Future Mask', fontsize=8)
                    axs[3].set_ylabel('Mask Value', fontsize=8)
                    axs[3].set_yticks([0, 1])

                    plt.xlabel('Timestep', fontsize=8)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])

                    save_filename = f"{sample_name_for_vis}_epoch{epoch}_batch{batch_idx}_autoregressive_model_masks.png"
                    save_path = os.path.join(vis_save_dir, save_filename)
                    plt.savefig(save_path)
                    plt.close(fig)
            except Exception as e:
                print(f"Error during mask visualization in GIMO_ADT_Autoregressive_Model: {e}")

        return total_loss, loss_dict 