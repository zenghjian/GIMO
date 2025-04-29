import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Actual GIMO components
from .pointnet_plus2 import PointNet2SemSegSSGShape
from .base_cross_model import PerceiveEncoder, PositionwiseFeedForward, PerceiveDecoder

class GIMO_ADT_Model(nn.Module):
    """
    GIMO model for ADT Object Motion Prediction.
    Uses the PointNet++ for scene encoding and Perceiver architecture for motion encoding.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Store history fraction for dynamic splitting
        self.history_fraction = config.history_fraction
        
        # Check if text embedding is disabled
        self.use_text_embedding = not getattr(config, 'no_text_embedding', False)
        
        # Still keep fixed trajectory_length for model definition
        self.sequence_length = config.trajectory_length 

        # --- Scene/Motion Encoding Components ---

        # 1. Motion Embedding
        self.motion_linear = nn.Linear(config.object_motion_dim, config.motion_hidden_dim)

        # 2. Point Cloud Encoder
        self.scene_encoder = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})

        # --- Category Encoding Components ---
        # Only initialize if text embedding is enabled
        if self.use_text_embedding:
            # Define token embedding dimension
            self.category_embed_dim = getattr(config, 'category_embed_dim', 64)
            
            # Max number of tokens per category string
            self.max_category_tokens = getattr(config, 'max_category_tokens', 30)
            
            # Vocabulary size for token embedding - keep simple for now
            self.vocab_size = getattr(config, 'vocab_size', 128)  # ASCII characters + special tokens
            
            # Create token embedding layer
            self.token_embedding = nn.Embedding(self.vocab_size, self.category_embed_dim)
            
            # Create category encoder using PerceiveEncoder to process tokens
            self.category_encoder = PerceiveEncoder(
                n_input_channels=self.category_embed_dim,
                n_latent=self.max_category_tokens,  # Match token sequence length
                n_latent_channels=getattr(config, 'category_latent_dim', 128),  # Output dimension for category encoding
                n_self_att_heads=getattr(config, 'category_n_heads', 4),
                n_self_att_layers=getattr(config, 'category_n_layers', 2),
                dropout=config.dropout
            )
            
            # Linear projector to match category features to same dimension as motion + scene
            self.category_projector = nn.Linear(
                getattr(config, 'category_latent_dim', 128),
                config.motion_latent_dim
            )
            
            # --- Cross-Modal Decoders (for motion-category interaction) ---
            # Similar to gaze_motion_decoder and motion_gaze_decoder in crossmodal_net.py
            self.motion_category_decoder = PerceiveDecoder(
                n_query_channels=config.motion_latent_dim,
                n_query=self.sequence_length,
                n_latent_channels=config.motion_latent_dim,  # Category features projected to this dimension
                dropout=config.dropout
            )
            
            self.category_motion_decoder = PerceiveDecoder(
                n_query_channels=config.motion_latent_dim,  # Category features projected to this dimension
                n_query=self.sequence_length,
                n_latent_channels=config.motion_latent_dim,
                dropout=config.dropout
            )

        # 3. Motion Encoder
        # Use fixed sequence length for latent definition
        self.motion_encoder = PerceiveEncoder(
            n_input_channels=config.motion_hidden_dim,  # Only motion features initially
            n_latent=self.sequence_length,
            n_latent_channels=config.motion_latent_dim,
            n_self_att_heads=config.motion_n_heads,
            n_self_att_layers=config.motion_n_layers,
            dropout=config.dropout
        )

        # 4. Output Encoder (renamed to match crossmodal_net.py)
        # Now with input size dependent on whether we use text embedding
        input_dim = config.motion_latent_dim + config.scene_feats_dim
        
        self.output_encoder = PerceiveEncoder(
            n_input_channels=input_dim,  # Scene features + (optionally fused) motion features
            n_latent=self.sequence_length,
            n_latent_channels=config.cross_hidden_dim,
            n_self_att_heads=config.cross_n_heads,
            n_self_att_layers=config.cross_n_layers,
            dropout=config.dropout
        )

        # 5. Embedding layer for processing combined features
        self.embedding_layer = PositionwiseFeedForward(
            input_dim,  # Scene + (optionally fused) motion features
            input_dim
        )

        # 6. Output Layer
        # Predicts per token
        self.outputlayer = nn.Linear(config.cross_hidden_dim, config.object_motion_dim)
    
    def tokenize_category(self, category_strings):
        """
        Simple tokenizer to convert category strings to token IDs.
        
        Args:
            category_strings: List of category strings [batch_size]
            
        Returns:
            Tensor of token IDs [batch_size, max_category_tokens]
        """
        if not self.use_text_embedding:
            # Return dummy tensor if text embedding is disabled
            return torch.zeros((len(category_strings), 1), dtype=torch.long, device=self.motion_encoder.latent.device)
            
        batch_size = len(category_strings)
        tokens = torch.zeros((batch_size, self.max_category_tokens), dtype=torch.long, device=self.token_embedding.weight.device)
        
        for b, cat_str in enumerate(category_strings):
            # Truncate if too long
            cat_str = cat_str[:self.max_category_tokens]
            
            # Convert characters to ASCII values (simple tokenization)
            for i, char in enumerate(cat_str):
                # Add +1 to avoid 0 (which we use as padding)
                tokens[b, i] = min(ord(char) + 1, self.vocab_size - 1)
                
        return tokens
    
    def encode_category(self, category_strings):
        """
        Encode category strings into feature vectors.
        
        Args:
            category_strings: List of category strings [batch_size]
            
        Returns:
            Tensor of category features [batch_size, category_latent_dim]
        """
        if not self.use_text_embedding:
            # Return zeros with the correct shape if text embedding is disabled
            return torch.zeros((len(category_strings), self.config.motion_latent_dim), 
                               device=self.motion_encoder.latent.device)
        
        # Tokenize the category strings
        tokens = self.tokenize_category(category_strings)
        
        # Embed the tokens
        token_embeddings = self.token_embedding(tokens)  # [B, max_tokens, embed_dim]
        
        # Encode the token embeddings
        encoded_categories = self.category_encoder(token_embeddings)  # [B, max_tokens, latent_dim]
        
        # Pool across tokens to get a single vector per category
        # Using mean pooling here, but could also use max pooling or attention
        category_features = torch.mean(encoded_categories, dim=1)  # [B, latent_dim]
        
        # Project to match motion latent dimensions
        category_features = self.category_projector(category_features)  # [B, motion_latent_dim]
        
        return category_features

    def forward(self, input_trajectory, point_cloud, object_categories):
        """
        Forward pass with architecture more closely aligned with crossmodal_net.py.
        Now includes object category information for improved prediction.

        Args:
            input_trajectory: Tensor [B, input_length, 3] - Only the input portion of trajectory
            point_cloud: Tensor [B, N_points, 3]
            object_categories: List of strings [B] - Category of each object (e.g., 'car', 'pedestrian')
            
        Returns:
            Tensor: Predicted full trajectory [B, sequence_length, 3] in standard mode,
                   or future trajectory [B, sequence_length-1, 3] in use_first_frame_only mode
        """
        batch_size = input_trajectory.shape[0]
        device = input_trajectory.device

        # 1. Embed input motion - No slicing needed as input is already the correct portion
        motion_feats = self.motion_linear(input_trajectory)  # [B, input_len, motion_hidden_dim]

        # 2. Encode point cloud (using PointNet2SemSegSSGShape)
        point_cloud_6d = point_cloud.repeat(1, 1, 2)  # [B, N, 3] -> [B, N, 6]
        scene_feats, scene_global_feats = self.scene_encoder(point_cloud_6d)  # [B, F, N_points], [B, F]

        # 3. Process motion with motion encoder
        motion_embedding = self.motion_encoder(motion_feats)
        # [B, sequence_length, motion_latent_dim]

        # 4. Encode object categories
        if self.use_text_embedding:
            category_features = self.encode_category(object_categories)  # [B, motion_latent_dim]
            
            # 5. Expand category features to match sequence length for cross-attention
            category_features_expanded = category_features.unsqueeze(1).repeat(1, motion_embedding.shape[1], 1)
            # [B, sequence_length, motion_latent_dim]
            
            # 6. Apply cross-attention between motion and category features
            # Motion attends to Category
            motion_with_category_context = self.motion_category_decoder(
                motion_embedding,  # Query [B, sequence_length, motion_latent_dim]
                category_features_expanded  # Latent [B, sequence_length, motion_latent_dim]
            )
            
            # Category attends to Motion
            category_with_motion_context = self.category_motion_decoder(
                category_features_expanded,  # Query [B, sequence_length, motion_latent_dim]
                motion_embedding  # Latent [B, sequence_length, motion_latent_dim]
            )
            
            # 7. Combine results from bidirectional cross-attention (element-wise addition)
            fused_motion_category = motion_with_category_context + category_with_motion_context
            # [B, sequence_length, motion_latent_dim]
        else:
            # Skip text embedding and cross-attention if disabled
            fused_motion_category = motion_embedding
            # [B, sequence_length, motion_latent_dim]

        # 8. Combine global scene features with fused motion-category features
        # Expand scene_global_feats to match sequence length
        scene_features_expanded = scene_global_feats.unsqueeze(1).repeat(1, motion_embedding.shape[1], 1)
        
        # Concatenate scene features with the fused motion-category features
        cross_modal_embedding = torch.cat([
            scene_features_expanded,   # [B, sequence_length, scene_feats_dim]
            fused_motion_category      # [B, sequence_length, motion_latent_dim]
        ], dim=2)
        # [B, sequence_length, scene_feats_dim + motion_latent_dim]

        # 9. Apply embedding layer
        cross_modal_embedding = self.embedding_layer(cross_modal_embedding)
        
        # 10. Final encoding with output encoder
        cross_modal_embedding = self.output_encoder(cross_modal_embedding)
        # [B, sequence_length, cross_hidden_dim]

        # 11. Predict trajectory for all tokens in the sequence
        all_predictions = self.outputlayer(cross_modal_embedding)

        # Return the full sequence prediction or only future if first_frame_only
        if self.config.use_first_frame_only:
            # Return only frames 1 to sequence_length-1
            return all_predictions[:, 1:, :] # Shape: [B, sequence_length-1, object_motion_dim]
        else:
            # Return the full sequence prediction
            return all_predictions # Shape: [B, sequence_length, object_motion_dim]

    def compute_loss(self, predictions, batch):
        """
        Computes the loss for the predicted trajectories using L1 norm, 
        incorporating dynamic history/future split based on attention mask.
        Losses are simplified to translation (position) and orientation components.
        
        When config.use_first_frame_only is True:
            - `predictions` tensor has shape [B, sequence_length-1, motion_dim]
            - Loss is calculated only on the future frames (GT indices 1 to N).

        Args:
            predictions: The model's output tensor. Shape depends on config.use_first_frame_only.
                         [B, sequence_length, motion_dim] if False.
                         [B, sequence_length-1, motion_dim] if True.
            batch: The batch dictionary from the DataLoader, containing ground truth.
                   Expected keys: 'full_poses' [B, sequence_length, motion_dim]
                                  'full_attention_mask' [B, sequence_length]

        Returns:
            torch.Tensor: The computed total loss value (scalar).
            dict: A dictionary containing individual loss components (batch averaged).
        """
        gt_full_poses = batch['full_poses'].to(predictions.device)
        gt_full_mask = batch['full_attention_mask'].to(predictions.device) # Shape [B, seq_len]
        batch_size = gt_full_poses.shape[0]
        device = predictions.device
        
        # Split poses into positions and orientations for separate loss calculation
        position_dim = 3  # Assuming first 3 dimensions are x, y, z
        
        # Extract ground truth positions and orientations
        gt_full_positions = gt_full_poses[..., :position_dim]
        gt_full_orientations = gt_full_poses[..., position_dim:]
        
        # Extract predicted positions and orientations
        pred_positions = predictions[..., :position_dim]
        pred_orientations = predictions[..., position_dim:]
        
        # --- Calculate Dynamic Split Lengths --- 
        actual_lengths = torch.sum(gt_full_mask, dim=1) # [B]
        # Ensure history length is at least 1 and doesn't exceed actual length, and handle dtypes
        # In use_first_frame_only mode, history length is effectively 1
        if self.config.use_first_frame_only:
            dynamic_history_lengths = torch.ones_like(actual_lengths).long() # History length is always 1
            # Clamp history length to actual length in case actual_length is 0 (empty trajectory)
            dynamic_history_lengths = torch.min(dynamic_history_lengths, actual_lengths.long())
        else:
            # Ensure min is also a tensor on the correct device
            min_val_tensor = torch.tensor(1, device=device)
            dynamic_history_lengths = torch.floor(actual_lengths * self.history_fraction).long().clamp(min=min_val_tensor, max=actual_lengths.long())
        # ---------------------------------------
        
        # Create masks for history and future based on dynamic lengths
        indices = torch.arange(self.sequence_length, device=device).unsqueeze(0) # [1, seq_len]
        dynamic_history_mask = (indices < dynamic_history_lengths.unsqueeze(1)).float() # [B, seq_len]
        dynamic_future_mask = (indices >= dynamic_history_lengths.unsqueeze(1)) * gt_full_mask # [B, seq_len]
        
        # Slice masks and GT if use_first_frame_only
        if self.config.use_first_frame_only:
            # Loss is calculated on GT indices 1 onwards, prediction indices 0 onwards
            gt_future_positions = gt_full_positions[:, 1:, :] # [B, seq_len-1, 3]
            gt_future_orientations = gt_full_orientations[:, 1:, :] # [B, seq_len-1, 3]
            dynamic_future_mask_for_loss = dynamic_future_mask[:, 1:] # [B, seq_len-1]
            
            # Predictions tensor already corresponds to GT indices 1 onwards
            pred_future_positions = pred_positions # [B, seq_len-1, 3]
            pred_future_orientations = pred_orientations # [B, seq_len-1, 3]
        else:
            # Use full GT and prediction for standard mode loss calculation
            gt_future_positions = gt_full_positions
            gt_future_orientations = gt_full_orientations
            pred_future_positions = pred_positions
            pred_future_orientations = pred_orientations
            dynamic_future_mask_for_loss = dynamic_future_mask
            
        # --- Calculate Translation Loss (Position) ---
        # This includes both history and future parts for standard mode,
        # but only future for first_frame_only mode
        loss_trans_per_coord = F.l1_loss(pred_future_positions, gt_future_positions, reduction='none') # [B, relevant_len, 3]
        loss_trans_per_point = torch.sum(loss_trans_per_coord, dim=-1) # [B, relevant_len]
        masked_loss_trans = loss_trans_per_point * dynamic_future_mask_for_loss # Apply mask [B, relevant_len]
        
        sum_loss_trans_per_seq = torch.sum(masked_loss_trans, dim=1) # [B]
        num_valid_points_per_seq = torch.sum(dynamic_future_mask_for_loss, dim=1) # [B]
        valid_trans_mask = num_valid_points_per_seq > 0
        mean_loss_trans_per_seq = torch.zeros_like(sum_loss_trans_per_seq)
        mean_loss_trans_per_seq[valid_trans_mask] = sum_loss_trans_per_seq[valid_trans_mask] / num_valid_points_per_seq[valid_trans_mask]
        mean_trans_loss = torch.sum(mean_loss_trans_per_seq) / torch.sum(valid_trans_mask) if torch.sum(valid_trans_mask) > 0 else torch.tensor(0.0, device=device)
        
        # --- Calculate Orientation Loss --- 
        loss_ori_per_coord = F.l1_loss(pred_future_orientations, gt_future_orientations, reduction='none') # [B, relevant_len, 3]
        loss_ori_per_point = torch.sum(loss_ori_per_coord, dim=-1) # [B, relevant_len]
        masked_loss_ori = loss_ori_per_point * dynamic_future_mask_for_loss # Apply mask [B, relevant_len]
        
        sum_loss_ori_per_seq = torch.sum(masked_loss_ori, dim=1) # [B]
        mean_loss_ori_per_seq = torch.zeros_like(sum_loss_ori_per_seq)
        mean_loss_ori_per_seq[valid_trans_mask] = sum_loss_ori_per_seq[valid_trans_mask] / num_valid_points_per_seq[valid_trans_mask]
        mean_ori_loss = torch.sum(mean_loss_ori_per_seq) / torch.sum(valid_trans_mask) if torch.sum(valid_trans_mask) > 0 else torch.tensor(0.0, device=device)
        
        # --- Calculate Reconstruction Loss (for input segment) ---
        if not self.config.use_first_frame_only and dynamic_history_lengths.max() > 0:
            # Get the history segment of ground truth
            hist_indices = torch.arange(self.sequence_length, device=device).unsqueeze(0)  # [1, seq_len]
            hist_mask = (hist_indices < dynamic_history_lengths.unsqueeze(1)) * gt_full_mask  # [B, seq_len]
            
            # Extract predictions and ground truth for history segment only
            hist_gt_poses = gt_full_poses  # [B, seq_len, 6]
            hist_pred_poses = predictions  # [B, seq_len, 6]
            
            # Calculate reconstruction loss for full poses (positions + orientations)
            loss_rec_per_coord = F.l1_loss(hist_pred_poses, hist_gt_poses, reduction='none')  # [B, seq_len, 6]
            loss_rec_per_point = torch.sum(loss_rec_per_coord, dim=-1)  # [B, seq_len]
            masked_loss_rec = loss_rec_per_point * hist_mask  # Apply mask [B, seq_len]
            
            sum_loss_rec_per_seq = torch.sum(masked_loss_rec, dim=1)  # [B]
            num_valid_hist_points_per_seq = torch.sum(hist_mask, dim=1)  # [B]
            valid_hist_mask = num_valid_hist_points_per_seq > 0
            mean_loss_rec_per_seq = torch.zeros_like(sum_loss_rec_per_seq)
            mean_loss_rec_per_seq[valid_hist_mask] = sum_loss_rec_per_seq[valid_hist_mask] / num_valid_hist_points_per_seq[valid_hist_mask]
            mean_rec_loss = torch.sum(mean_loss_rec_per_seq) / torch.sum(valid_hist_mask) if torch.sum(valid_hist_mask) > 0 else torch.tensor(0.0, device=device)
        else:
            # No history to reconstruct
            mean_rec_loss = torch.tensor(0.0, device=device)
        
        # --- Apply weights ---
        # Use simplified weights: lambda_trans, lambda_ori, lambda_rec
        lambda_trans = getattr(self.config, 'lambda_trans', 1.0)
        lambda_ori = getattr(self.config, 'lambda_ori', 1.0)
        lambda_rec = getattr(self.config, 'lambda_rec', 1.0)  # Single weight for reconstruction loss
        
        weighted_trans_loss = lambda_trans * mean_trans_loss
        weighted_ori_loss = lambda_ori * mean_ori_loss
        weighted_rec_loss = lambda_rec * mean_rec_loss
        
        # --- Calculate Destination Loss (终点损失) ---
        # Find the last valid point for each sequence in the batch
        batch_size = gt_full_poses.shape[0]
        last_valid_indices = (torch.sum(gt_full_mask, dim=1) - 1).long().clamp(min=0)  # [B]
        batch_indices = torch.arange(batch_size, device=device)
        
        # Extract the last valid GT and predicted positions
        last_gt_poses = gt_full_poses[batch_indices, last_valid_indices]  # [B, 6]
        
        # Handle the first_frame_only case differently for predictions
        if self.config.use_first_frame_only:
            # In this mode, prediction indices are shifted by 1 relative to GT
            pred_indices = torch.clamp(last_valid_indices - 1, min=0)
            if predictions.shape[1] > 1:  # Make sure predictions has enough frames
                last_pred_poses = predictions[batch_indices, pred_indices]  # [B, 6]
            else:
                last_pred_poses = predictions[:, -1]  # Use last available prediction
        else:
            last_pred_poses = predictions[batch_indices, last_valid_indices]  # [B, 6]
        
        # Calculate destination loss for both positions and orientations
        dest_loss_per_coord = F.l1_loss(last_pred_poses, last_gt_poses, reduction='none')  # [B, 6]
        
        # Split into position and orientation components
        dest_trans_loss_per_seq = torch.sum(dest_loss_per_coord[:, :position_dim], dim=1)  # [B]
        dest_ori_loss_per_seq = torch.sum(dest_loss_per_coord[:, position_dim:], dim=1)  # [B]
        
        # Apply mask to handle invalid sequences
        valid_dest_mask = last_valid_indices > 0  # Sequences with at least one valid point
        dest_trans_loss = torch.sum(dest_trans_loss_per_seq * valid_dest_mask.float()) / torch.sum(valid_dest_mask.float()) if torch.sum(valid_dest_mask) > 0 else torch.tensor(0.0, device=device)
        dest_ori_loss = torch.sum(dest_ori_loss_per_seq * valid_dest_mask.float()) / torch.sum(valid_dest_mask.float()) if torch.sum(valid_dest_mask) > 0 else torch.tensor(0.0, device=device)
        
        weighted_dest_trans_loss = lambda_trans * dest_trans_loss 
        weighted_dest_ori_loss = lambda_ori * dest_ori_loss 
        
        # Total loss
        total_loss = weighted_trans_loss + weighted_ori_loss + weighted_rec_loss + weighted_dest_trans_loss + weighted_dest_ori_loss
                     
        # Store individual batch-averaged losses for logging/debugging
        loss_dict = {
            'total_loss': total_loss,
            'trans_loss': weighted_trans_loss,
            'ori_loss': weighted_ori_loss,
            'rec_loss': weighted_rec_loss,
            'dest_trans_loss': weighted_dest_trans_loss,
            'dest_ori_loss': weighted_dest_ori_loss
        }
            
        return total_loss, loss_dict

# Example usage (for testing structure)
if __name__ == '__main__':
    # Import config (assuming it's runnable)
    import sys
    sys.path.append('../') # Add project root to path
    from config.adt_config import ADTObjectMotionConfig
    
    # --- Add imports for PointNet dependencies if necessary ---
    # This might be needed if running this script directly requires ops
    try:
        from pointnet2_ops import pointnet2_utils
    except ImportError:
        print("Warning: pointnet2_ops not found. May be needed if running model script directly.")
        # If running train/test scripts that handle imports/env setup, this might not be an issue.
    # --------------------------------------------------------

    config = ADTObjectMotionConfig().get_configs()
    
    # Override sequence lengths for this test if needed
    config.history_length = 62
    config.future_length = 38
    config.object_motion_dim = 3
    config.point_cloud_dim = 3
    config.batch_size = 4 # Example batch size
    config.sample_points = 1024 # Example point cloud size
    
    # Test both with and without text embedding
    for no_text_embedding in [False, True]:
        # Set the text embedding flag
        config.no_text_embedding = no_text_embedding
        print(f"\n--- Testing with {'NO' if no_text_embedding else ''} text embedding ---")
        
        # Create dummy input data
        past_traj_dummy = torch.randn(config.batch_size, config.history_length, config.object_motion_dim)
        point_cloud_dummy = torch.randn(config.batch_size, config.sample_points, config.point_cloud_dim)
        object_categories_dummy = ["car", "pedestrian", "bicycle", "car"]  # Example category strings
        
        # Instantiate model
        model = GIMO_ADT_Model(config)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Text embedding enabled: {model.use_text_embedding}")
    
        # Forward pass (updated to include object_categories)
        output = model(past_traj_dummy, point_cloud_dummy, object_categories_dummy)
        
        print(f"Input past trajectory shape: {past_traj_dummy.shape}")
        print(f"Input point cloud shape: {point_cloud_dummy.shape}")
        print(f"Input object categories: {object_categories_dummy}")
        print(f"Output predicted future shape: {output.shape}")
        
        # Check output shape
        expected_shape = (config.batch_size, config.sequence_length, config.object_motion_dim)
        assert output.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, Got {output.shape}"
        print("Output shape check passed.") 