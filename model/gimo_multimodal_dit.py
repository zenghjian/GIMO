#!/usr/bin/env python3
"""
GIMO Multimodal Diffusion Transformer (DiT) for trajectory generation.

This implementation combines:
1. GIMO's perceiver-based multimodal conditioning (motion + scene + category + semantic bbox)
2. DiT architecture for trajectory diffusion
3. Conditional generation using multimodal latent arrays

Based on:
- DiT paper: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
- GIMO multimodal architecture
- Aria World Gaussians trajectory diffusion implementation
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from einops import rearrange

# Import GIMO components
from .pointnet_plus2 import PointNet2SemSegSSGShape, PointNet, MyFPModule
from .base_cross_model import PerceiveEncoder, PositionwiseFeedForward, PerceiveDecoder
import clip

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimestepEmbedder(nn.Module):
    """Embeds diffusion timesteps into a higher dimensional space."""
    
    def __init__(self, hidden_dim: int, time_embed_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_embed(timesteps)


class PatchEmbed1D(nn.Module):
    """1D version of PatchEmbed for trajectory data."""
    
    def __init__(self, seq_length: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = seq_length // patch_size
        
        # Check if seq_length is divisible by patch_size
        assert seq_length % patch_size == 0, f"Sequence length {seq_length} not divisible by patch size {patch_size}"
        
        # Projection from patch to embedding
        self.proj = nn.Linear(in_channels * patch_size, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, L, C) where L is sequence length, C is channels
        B, L, C = x.shape
        assert L == self.seq_length, f"Input sequence length ({L}) doesn't match model sequence length ({self.seq_length})"
        
        # Reshape into patches
        x = x.reshape(B, self.num_patches, self.patch_size * C)
        
        # Project patches to embedding dimension
        x = self.proj(x)
        
        return x


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization with conditioning from embeddings."""
    
    def __init__(self, hidden_dim: int, cond_embed_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_embed_dim = cond_embed_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale_shift = nn.Linear(cond_embed_dim, 2 * hidden_dim)
        
    def forward(self, x, cond_emb):
        """
        Args:
            x: Input tensor [B, num_patches, hidden_dim]
            cond_emb: Conditioning embedding [B, cond_embed_dim]
        """
        # Layer normalization
        x = self.norm(x)
        
        # Calculate scale and shift from conditioning
        scale_shift = self.scale_shift(cond_emb)
        scale_shift = scale_shift.unsqueeze(1)  # [B, 1, 2*hidden_dim]
        
        # Split into scale and shift
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.expand_as(x)
        shift = shift.expand_as(x)
        
        # Apply scale and shift
        return x * (1 + scale) + shift


class MultimodalConditionedDiTBlock(nn.Module):
    """DiT Block with multimodal conditioning from GIMO perceiver features."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cond_embed_dim: int = 640
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Self-attention with multimodal conditioning
        self.norm1 = AdaLayerNorm(hidden_dim, cond_embed_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Cross-attention to multimodal features
        self.norm_cross = AdaLayerNorm(hidden_dim, cond_embed_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Feed-forward network
        self.norm2 = AdaLayerNorm(hidden_dim, cond_embed_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
    
    def forward(self, x, cond_emb, multimodal_features=None):
        """
        Args:
            x: Input tensor [B, num_patches, hidden_dim]
            cond_emb: Conditioning embedding (time + other conditioning) [B, cond_embed_dim]
            multimodal_features: Multimodal latent array from GIMO perceiver [B, perceiver_latent_size, feature_dim] - Fixed latent size
        """
        # Self-attention
        residual = x
        x = self.norm1(x, cond_emb)
        x = self.self_attn(x, x, x)[0]
        x = x + residual
        
        # Cross-attention to multimodal features (if provided)
        if multimodal_features is not None:
            residual = x
            x = self.norm_cross(x, cond_emb)
            
            # Project multimodal features to hidden_dim if needed
            if multimodal_features.shape[-1] != self.hidden_dim:
                # Simple linear projection
                if not hasattr(self, 'multimodal_proj'):
                    self.multimodal_proj = nn.Linear(multimodal_features.shape[-1], self.hidden_dim).to(x.device)
                multimodal_features = self.multimodal_proj(multimodal_features)
            
            x = self.cross_attn(
                query=x,
                key=multimodal_features,
                value=multimodal_features
            )[0]
            x = x + residual
        
        # Feed-forward network
        residual = x
        x = self.norm2(x, cond_emb)
        x = self.ffn(x)
        x = x + residual
        
        return x


class FinalLayer(nn.Module):
    """Final layer for DiT that outputs the denoised trajectory."""
    
    def __init__(self, hidden_dim: int, out_channels: int, cond_embed_dim: int):
        super().__init__()
        self.norm_final = AdaLayerNorm(hidden_dim, cond_embed_dim)
        self.linear = nn.Linear(hidden_dim, out_channels)
    
    def forward(self, x, cond_emb):
        x = self.norm_final(x, cond_emb)
        x = self.linear(x)
        return x


class BBoxEmbedder(nn.Module):
    """Embeds bounding box information for scene conditioning."""
    
    def __init__(self, input_dim=12, output_dim=640, hidden_dim=128, num_heads=4, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # Projection layers
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Self-attention for object relationships
        if use_attention:
            self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        
    def forward(self, x, mask=None):
        if x is None:
            return None
        
        batch_size, num_objects, input_dim = x.shape
        
        # Project each bounding box
        x_flat = x.reshape(-1, input_dim)
        bbox_feats = self.proj(x_flat)
        bbox_feats = bbox_feats.reshape(batch_size, num_objects, -1)
        
        # Apply self-attention
        if self.use_attention:
            if mask is not None:
                bool_mask = (mask == 0)
                attn_out, _ = self.self_attn(
                    bbox_feats, bbox_feats, bbox_feats,
                    key_padding_mask=bool_mask
                )
            else:
                attn_out, _ = self.self_attn(bbox_feats, bbox_feats, bbox_feats)
            bbox_feats = attn_out
        
        # Apply mask and global pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(bbox_feats)
            bbox_feats = bbox_feats * mask_expanded
            valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled_feats = bbox_feats.sum(dim=1) / valid_counts
        else:
            pooled_feats = bbox_feats.mean(dim=1)
        
        # Final projection
        result = self.final_proj(pooled_feats)
        return result


class CategoryEmbedder(nn.Module):
    """Embeds object categories using CLIP."""
    
    def __init__(self, output_dim=640, clip_model_name="ViT-B/32"):
        super().__init__()
        
        # Load CLIP model
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Get CLIP embedding dimension
        if clip_model_name == "ViT-B/32":
            self.clip_embed_dim = 512
        elif clip_model_name == "ViT-B/16":
            self.clip_embed_dim = 512
        elif clip_model_name == "ViT-L/14":
            self.clip_embed_dim = 768
        else:
            self.clip_embed_dim = 512
        
        # Projection to match required dimension
        self.proj = nn.Sequential(
            nn.Linear(self.clip_embed_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, categories):
        if categories is None:
            return None
        
        device = next(self.parameters()).device
        
        # Move CLIP model to correct device
        if next(self.clip_model.parameters()).device != device:
            self.clip_model = self.clip_model.to(device)
        
        # Handle empty or invalid categories
        processed_categories = []
        for cat in categories:
            if cat is None or cat == '' or cat == 'unknown':
                processed_categories.append("object")
            else:
                processed_categories.append(str(cat))
        
        # Encode with CLIP
        try:
            with torch.no_grad():
                text_tokens = clip.tokenize(processed_categories, truncate=True).to(device)
                embeddings = self.clip_model.encode_text(text_tokens).float()
        except Exception as e:
            print(f"Warning: CLIP encoding failed: {e}. Using fallback embeddings.")
            batch_size = len(categories)
            embeddings = torch.zeros(batch_size, self.clip_embed_dim, dtype=torch.float32, device=device)
        
        # Project to required dimension
        return self.proj(embeddings)


class GIMOMultimodalConditioning(nn.Module):
    """
    GIMO-based multimodal conditioning module.
    Uses perceiver to compress different modalities into a latent array for DiT conditioning.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Store feature dimensions
        self.motion_hidden_dim = config.motion_hidden_dim
        self.scene_feats_dim = config.scene_feats_dim
        self.motion_latent_dim = config.motion_latent_dim
        self.category_embed_dim = config.category_embed_dim
        self.semantic_bbox_embed_dim = getattr(config, 'semantic_bbox_embed_dim', 256)
        self.perceiver_latent_size = getattr(config, 'dit_perceiver_latent_size', 64)  # Configurable latent size
        
        # Control flags
        self.use_bbox = not getattr(config, 'no_bbox', False)
        self.use_scene = not getattr(config, 'no_scene', False)
        self.use_text_embedding = not getattr(config, 'no_text_embedding', False)
        self.use_semantic_bbox = not getattr(config, 'no_semantic_bbox', False)
        
        # Motion embedding
        self.motion_linear = nn.Linear(config.object_motion_dim, config.motion_hidden_dim)
        
        # Scene encoder (PointNet++)
        self.scene_encoder = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})
        
        # Semantic BBox Embedder
        if self.use_semantic_bbox:
            self.semantic_bbox_embedder = BBoxEmbedder(
                input_dim=12,
                output_dim=config.semantic_bbox_embed_dim,
                hidden_dim=getattr(config, 'semantic_bbox_hidden_dim', 128),
                num_heads=getattr(config, 'semantic_bbox_num_heads', 4),
                use_attention=getattr(config, 'semantic_bbox_use_attention', True)
            )
        
        # Bounding box processing
        if self.use_bbox:
            self.fp_layer = MyFPModule()
            self.bbox_pointnet = PointNet(config.scene_feats_dim)
        
        # Motion-bbox encoder (first perceiver)
        if self.use_bbox:
            self.motion_bbox_encoder = PerceiveEncoder(
                n_input_channels=config.motion_hidden_dim + config.scene_feats_dim,
                n_latent=self.perceiver_latent_size,  # Configurable latent size
                n_latent_channels=config.motion_latent_dim,
                n_self_att_heads=config.motion_n_heads,
                n_self_att_layers=config.motion_n_layers,
                dropout=config.dropout
            )
        else:
            self.motion_bbox_encoder = PerceiveEncoder(
                n_input_channels=config.motion_hidden_dim,
                n_latent=self.perceiver_latent_size,  # Configurable latent size
                n_latent_channels=config.motion_latent_dim,
                n_self_att_heads=config.motion_n_heads,
                n_self_att_layers=config.motion_n_layers,
                dropout=config.dropout
            )
        
        # Category embedder
        if self.use_text_embedding:
            self.category_embedder = CategoryEmbedder(
                output_dim=config.category_embed_dim,
                clip_model_name=getattr(config, 'clip_model_name', "ViT-B/32")
            )
        
        # Calculate final embedding dimension
        embed_dim = config.motion_latent_dim
        if self.use_scene:
            embed_dim += config.scene_feats_dim
        if self.use_text_embedding:
            embed_dim += config.category_embed_dim
        if self.use_semantic_bbox:
            embed_dim += config.semantic_bbox_embed_dim
        
        # Final embedding layer
        self.embedding_layer = PositionwiseFeedForward(
            d_in=embed_dim,
            d_hid=embed_dim,
            dropout=config.dropout
        )
        
        # Final perceiver encoder for multimodal fusion
        self.multimodal_encoder = PerceiveEncoder(
            n_input_channels=embed_dim,
            n_latent=self.perceiver_latent_size,  # Configurable latent size
            n_latent_channels=getattr(config, 'dit_cond_dim', 512),  # Output dimension for DiT conditioning
            n_self_att_heads=getattr(config, 'dit_cond_heads', 8),
            n_self_att_layers=getattr(config, 'dit_cond_layers', 3),
            dropout=config.dropout
        )
    
    def forward(self, input_trajectory, point_cloud, bounding_box_corners=None, 
                object_category_ids=None, semantic_bbox_info=None, semantic_bbox_mask=None):
        """
        Generate multimodal conditioning features.
        
        Args:
            input_trajectory: [B, input_len, 9] - input trajectory
            point_cloud: [B, num_points, 3] - scene point cloud
            bounding_box_corners: [B, input_len, 8, 3] - bbox corners (optional)
            object_category_ids: [B] - category strings (optional)
            semantic_bbox_info: [B, max_bboxes, 12] - scene bbox info (optional)
            semantic_bbox_mask: [B, max_bboxes] - scene bbox mask (optional)
        
        Returns:
            torch.Tensor: Multimodal conditioning features [B, perceiver_latent_size, dit_cond_dim] - Fixed latent size
        """
        batch_size = input_trajectory.shape[0]
        input_length = input_trajectory.shape[1]
        
        # Process motion
        f_m = self.motion_linear(input_trajectory)  # [B, input_len, motion_hidden_dim]
        
        # Process scene point cloud
        point_cloud_6d = torch.cat([point_cloud, point_cloud], dim=2)  # [B, N, 6]
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            scene_feats_per_point, scene_global_feats = self.scene_encoder(point_cloud_6d)
        
        # Process semantic bbox (if enabled)
        semantic_bbox_embeddings = None
        if self.use_semantic_bbox and semantic_bbox_info is not None:
            semantic_bbox_embeddings = self.semantic_bbox_embedder(semantic_bbox_info, semantic_bbox_mask)
        
        # Process trajectory-specific bboxes (if enabled)
        if self.use_bbox and bounding_box_corners is not None:
            num_bbox_corners = bounding_box_corners.shape[2]
            
            # Reshape for feature propagation
            point_cloud_repeated = point_cloud.unsqueeze(1).repeat(1, input_length, 1, 1)
            point_cloud_for_fp = point_cloud_repeated.reshape(batch_size * input_length, -1, 3).contiguous()
            
            scene_feats_repeated = scene_feats_per_point.unsqueeze(2).repeat(1, 1, input_length, 1)
            scene_feats_for_fp = scene_feats_repeated.permute(0, 2, 1, 3).reshape(
                batch_size * input_length, self.scene_feats_dim, -1
            ).contiguous()
            
            bbox_corners_for_fp = bounding_box_corners.reshape(batch_size * input_length, num_bbox_corners, 3).contiguous()
            
            # Feature propagation
            propagated_feats = self.fp_layer(
                unknown=bbox_corners_for_fp,
                known=point_cloud_for_fp,
                known_feats=scene_feats_for_fp
            )
            
            # Process with PointNet
            f_s_b_per_ts = self.bbox_pointnet(propagated_feats)
            f_s_b = f_s_b_per_ts.reshape(batch_size, input_length, self.scene_feats_dim)
            
            # Fuse motion and bbox features
            fused_motion_bbox_input = torch.cat([f_m, f_s_b], dim=2)
        else:
            fused_motion_bbox_input = f_m
        
        # First perceiver encoding (motion + bbox)
        encoded_motion_bbox = self.motion_bbox_encoder(fused_motion_bbox_input)
        
        # Generate category embeddings
        category_embeddings_expanded = None
        if self.use_text_embedding and object_category_ids is not None:
            category_embeddings = self.category_embedder(object_category_ids)
            category_embeddings_expanded = category_embeddings.unsqueeze(1).repeat(1, self.perceiver_latent_size, 1)  # Use configurable latent size
        
        # Prepare features for fusion
        features_to_fuse = []
        
        # Always include motion features
        features_to_fuse.append(encoded_motion_bbox)
        
        # Add scene features
        if self.use_scene:
            scene_global_feats_expanded = scene_global_feats.unsqueeze(1).repeat(1, self.perceiver_latent_size, 1)  # Use configurable latent size
            features_to_fuse.append(scene_global_feats_expanded)
        
        # Add category features
        if category_embeddings_expanded is not None:
            features_to_fuse.append(category_embeddings_expanded)
        
        # Add semantic bbox features
        if semantic_bbox_embeddings is not None:
            semantic_bbox_embeddings_expanded = semantic_bbox_embeddings.unsqueeze(1).repeat(1, self.perceiver_latent_size, 1)  # Use configurable latent size
            features_to_fuse.append(semantic_bbox_embeddings_expanded)
        
        # Concatenate all features
        final_fused_input = torch.cat(features_to_fuse, dim=2)
        
        # Apply embedding layer
        cross_modal_embedding = self.embedding_layer(final_fused_input)
        
        # Final perceiver encoding for DiT conditioning
        multimodal_conditioning = self.multimodal_encoder(cross_modal_embedding)
        
        return multimodal_conditioning


class GIMO_MultimodalDiT(nn.Module):
    """
    GIMO Multimodal Diffusion Transformer.
    Combines GIMO's multimodal conditioning with DiT architecture for trajectory generation.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # DiT architecture parameters
        self.input_dim = config.object_motion_dim  # 9 for [x,y,z,r6d]
        self.seq_length = config.trajectory_length
        self.patch_size = getattr(config, 'dit_patch_size', 8)
        self.hidden_dim = getattr(config, 'dit_hidden_dim', 384)
        self.depth = getattr(config, 'dit_depth', 12)
        self.num_heads = getattr(config, 'dit_num_heads', 6)
        self.mlp_ratio = getattr(config, 'dit_mlp_ratio', 4.0)
        self.time_embed_dim = getattr(config, 'dit_time_embed_dim', 640)
        self.cond_embed_dim = getattr(config, 'dit_cond_embed_dim', 640)
        
        # Multimodal conditioning module (GIMO-based)
        self.multimodal_conditioning = GIMOMultimodalConditioning(config)
        
        # Patch embedding for trajectory input
        self.patch_embed = PatchEmbed1D(
            seq_length=self.seq_length,
            patch_size=self.patch_size,
            in_channels=self.input_dim,
            embed_dim=self.hidden_dim
        )
        
        # Timestep embedding
        self.time_embedder = TimestepEmbedder(
            hidden_dim=self.hidden_dim,
            time_embed_dim=self.time_embed_dim
        )
        
        # Position embeddings for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.hidden_dim))
        
        # DiT blocks with multimodal conditioning
        self.blocks = nn.ModuleList([
            MultimodalConditionedDiTBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                cond_embed_dim=self.cond_embed_dim
            )
            for _ in range(self.depth)
        ])
        
        # Final output layer
        self.final_layer = FinalLayer(
            hidden_dim=self.hidden_dim,
            out_channels=self.patch_size * self.input_dim,
            cond_embed_dim=self.cond_embed_dim
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize other weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to sequence format."""
        # x shape: [B, num_patches, patch_size * input_dim]
        B, num_patches, patch_dim = x.shape
        
        # Reshape to [B, num_patches, patch_size, input_dim]
        x = x.reshape(B, num_patches, self.patch_size, self.input_dim)
        
        # Rearrange to [B, num_patches * patch_size, input_dim]
        x = x.reshape(B, num_patches * self.patch_size, self.input_dim)
        
        return x
    
    def forward(self, x, timesteps, input_trajectory, point_cloud, 
                bounding_box_corners=None, object_category_ids=None,
                semantic_bbox_info=None, semantic_bbox_mask=None):
        """
        Forward pass through the multimodal DiT.
        
        Args:
            x: Noisy trajectory tensor [B, seq_length, input_dim]
            timesteps: Diffusion timesteps [B]
            input_trajectory: Input trajectory for conditioning [B, input_len, 9]
            point_cloud: Scene point cloud [B, num_points, 3]
            bounding_box_corners: Trajectory bbox corners [B, input_len, 8, 3] (optional)
            object_category_ids: Category strings [B] (optional)
            semantic_bbox_info: Scene bbox info [B, max_bboxes, 12] (optional)
            semantic_bbox_mask: Scene bbox mask [B, max_bboxes] (optional)
        
        Returns:
            torch.Tensor: Denoised trajectory [B, seq_length, input_dim]
        """
        # Generate multimodal conditioning features
        multimodal_features = self.multimodal_conditioning(
            input_trajectory=input_trajectory,
            point_cloud=point_cloud,
            bounding_box_corners=bounding_box_corners,
            object_category_ids=object_category_ids,
            semantic_bbox_info=semantic_bbox_info,
            semantic_bbox_mask=semantic_bbox_mask
        )  # [B, perceiver_latent_size, dit_cond_dim] - Fixed latent size
        
        # Generate timestep embeddings
        time_emb = self.time_embedder(timesteps)  # [B, time_embed_dim]
        
        # Combine time and multimodal conditioning
        # Simple approach: project multimodal features to condition embedding dimension
        if multimodal_features.shape[-1] != self.cond_embed_dim:
            if not hasattr(self, 'cond_proj'):
                self.cond_proj = nn.Linear(multimodal_features.shape[-1], self.cond_embed_dim).to(x.device)
            multimodal_cond = self.cond_proj(multimodal_features.mean(dim=1))  # [B, cond_embed_dim]
        else:
            multimodal_cond = multimodal_features.mean(dim=1)  # [B, cond_embed_dim]
        
        # Combine time and multimodal conditioning
        combined_cond = time_emb + multimodal_cond  # [B, cond_embed_dim]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, hidden_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply DiT blocks with multimodal conditioning
        for block in self.blocks:
            x = block(x, combined_cond, multimodal_features)
        
        # Final layer
        x = self.final_layer(x, combined_cond)
        
        # Convert patches back to sequence
        x = self.unpatchify(x)
        
        return x


class DiffusionTrainer:
    """Trainer for GIMO Multimodal DiT using DDPM."""
    
    def __init__(
        self,
        model: nn.Module,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        timesteps: int = 1000,
        loss_type: str = "l2",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initialize the diffusion trainer.
        
        Args:
            model: The GIMO_MultimodalDiT model
            beta_start: Starting noise schedule value
            beta_end: Ending noise schedule value  
            timesteps: Number of diffusion timesteps
            loss_type: Loss function type ("l1" or "l2")
            device: Device for training
        """
        self.model = model
        self.device = device
        self.timesteps = timesteps
        self.loss_type = loss_type
        
        # Create noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
    def extract(self, a, t, x_shape):
        """Extract values from a 1-D numpy array for a batch of indices."""
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0) - the forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, attention_mask=None, **conditioning_inputs):
        """
        Compute training losses for the diffusion model.
        
        Args:
            x_start: Clean trajectory data [B, seq_length, input_dim]
            t: Timesteps [B]
            attention_mask: Attention mask for valid trajectory points [B, seq_length]
            **conditioning_inputs: Conditioning inputs for multimodal features
        
        Returns:
            torch.Tensor: Loss value
        """
        # Generate noise
        noise = torch.randn_like(x_start)
        
        # Apply mask to noise if provided
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x_start)
            noise = noise * mask_expanded
        
        # Forward diffusion (add noise)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Apply mask to noisy input if provided
        if attention_mask is not None:
            x_noisy = x_noisy * mask_expanded
        
        # Predict noise using the model
        predicted_noise = self.model(x_noisy, t, **conditioning_inputs)
        
        # Compute loss
        if self.loss_type == 'l1':
            loss_fn = F.l1_loss
        elif self.loss_type == 'l2':
            loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if attention_mask is not None:
            # Masked loss computation
            loss_per_element = loss_fn(predicted_noise, noise, reduction='none')
            masked_loss = loss_per_element * mask_expanded
            loss = masked_loss.sum() / mask_expanded.sum()
        else:
            loss = loss_fn(predicted_noise, noise)
        
        return loss
    
    def train_step(self, batch, optimizer):
        """
        Perform one training step.
        
        Args:
            batch: Batch of training data
            optimizer: Optimizer for model parameters
        
        Returns:
            float: Loss value
        """
        # Extract trajectory data
        x_start = batch['full_poses'].float().to(self.device)  # [B, seq_length, input_dim]
        attention_mask = batch.get('full_attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.float().to(self.device)
        
        # Extract conditioning inputs
        conditioning_inputs = {}
        
        # Input trajectory (history part)
        if 'input_trajectory' in batch:
            conditioning_inputs['input_trajectory'] = batch['input_trajectory'].float().to(self.device)
            input_length = conditioning_inputs['input_trajectory'].shape[1]
        else:
            # Extract input trajectory from full trajectory using history fraction
            batch_size = x_start.shape[0]
            if attention_mask is not None:
                actual_lengths = attention_mask.sum(dim=1).int()
                history_lengths = (actual_lengths * self.model.config.history_fraction).floor().long().clamp(min=1)
                max_history_length = history_lengths.max().item()
                conditioning_inputs['input_trajectory'] = x_start[:, :max_history_length, :]
                input_length = max_history_length
            else:
                history_length = int(x_start.shape[1] * getattr(self.model.config, 'history_fraction', 0.3))
                conditioning_inputs['input_trajectory'] = x_start[:, :history_length, :]
                input_length = history_length
        
        # Scene point cloud
        if 'point_cloud' in batch:
            conditioning_inputs['point_cloud'] = batch['point_cloud'].float().to(self.device)
        
        # Bounding box corners - truncate to match input_trajectory length
        if 'bbox_corners' in batch:
            full_bbox_corners = batch['bbox_corners'].float().to(self.device)
            # Truncate bbox_corners to match input_trajectory length
            conditioning_inputs['bounding_box_corners'] = full_bbox_corners[:, :input_length, :, :]
        
        # Object categories
        if 'object_category' in batch:
            conditioning_inputs['object_category_ids'] = batch['object_category']
        
        # Semantic bbox info
        if 'scene_bbox_info' in batch:
            conditioning_inputs['semantic_bbox_info'] = batch['scene_bbox_info'].float().to(self.device)
        if 'scene_bbox_mask' in batch:
            conditioning_inputs['semantic_bbox_mask'] = batch['scene_bbox_mask'].float().to(self.device)
        
        # Random timesteps
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # Compute loss
        loss = self.p_losses(x_start, t, attention_mask, **conditioning_inputs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def p_sample(self, x, t, **conditioning_inputs):
        """Sample from p(x_{t-1} | x_t) - single reverse diffusion step."""
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(1.0 / torch.sqrt(self.alphas), t, x.shape)
        
        # Predict noise
        predicted_noise = self.model(x, t, **conditioning_inputs)
        
        # Compute mean of q(x_{t-1} | x_t, x_0)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, batch_size, seq_length, channels, **conditioning_inputs):
        """
        Generate samples using reverse diffusion process.
        
        Args:
            batch_size: Number of samples to generate
            seq_length: Length of trajectory sequences
            channels: Number of channels (input_dim)
            **conditioning_inputs: Conditioning inputs for generation
        
        Returns:
            torch.Tensor: Generated trajectory samples [B, seq_length, channels]
        """
        # Start from random noise
        x = torch.randn(batch_size, seq_length, channels, device=self.device)
        
        # Apply reverse diffusion
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, **conditioning_inputs)
        
        return x
    
    @torch.no_grad()
    def complete_trajectory(self, input_trajectory, **conditioning_inputs):
        """
        Complete a trajectory given an input prefix.
        
        Args:
            input_trajectory: Input trajectory prefix [B, input_len, channels]
            **conditioning_inputs: Additional conditioning inputs
        
        Returns:
            torch.Tensor: Completed trajectory [B, seq_length, channels]
        """
        batch_size, input_len, channels = input_trajectory.shape
        seq_length = self.model.seq_length
        
        # Ensure conditioning inputs include the input trajectory
        conditioning_inputs['input_trajectory'] = input_trajectory
        
        # Generate the full trajectory
        generated = self.sample(batch_size, seq_length, channels, **conditioning_inputs)
        
        return generated 