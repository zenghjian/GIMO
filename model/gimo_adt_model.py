import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Actual GIMO components
from .pointnet_plus2 import PointNet2SemSegSSGShape
from .base_cross_model import PerceiveEncoder, PositionwiseFeedForward

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
        
        # Still keep fixed trajectory_length for model definition
        self.sequence_length = config.trajectory_length 

        # --- Scene/Motion Encoding Components ---

        # 1. Motion Embedding
        self.motion_linear = nn.Linear(config.object_motion_dim, config.motion_hidden_dim)

        # 2. Point Cloud Encoder
        self.scene_encoder = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})

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
        # Analogous to output_encoder in crossmodal_net.py
        # Use fixed sequence length for latent definition
        self.output_encoder = PerceiveEncoder(
            n_input_channels=config.motion_latent_dim + config.scene_feats_dim,  # Motion latent + scene features
            n_latent=self.sequence_length,
            n_latent_channels=config.cross_hidden_dim,
            n_self_att_heads=config.cross_n_heads,
            n_self_att_layers=config.cross_n_layers,
            dropout=config.dropout
        )

        # 5. Embedding layer for processing combined features (analogous to embedding_layer in crossmodal_net.py)
        self.embedding_layer = PositionwiseFeedForward(
            config.motion_latent_dim + config.scene_feats_dim,
            config.motion_latent_dim + config.scene_feats_dim
        )

        # 6. Output Layer (analogous to outputlayer in crossmodal_net.py)
        # Predicts per token
        self.outputlayer = nn.Linear(config.cross_hidden_dim, config.object_motion_dim)

    def forward(self, full_trajectory, point_cloud):
        """
        Forward pass with architecture more closely aligned with crossmodal_net.py.
        Takes full trajectory as input.

        Args:
            full_trajectory: Tensor [B, sequence_length, 3]
            point_cloud: Tensor [B, N_points, 3]
            
        Returns:
            Tensor: Predicted full trajectory [B, sequence_length, 3]
        """
        batch_size = full_trajectory.shape[0]
        device = full_trajectory.device

        # --- Prepare Motion Input --- 
        if self.config.use_first_frame_only:
             # Use only the first frame as input
             motion_input = full_trajectory[:, 0:1, :] # Shape: [B, 1, 3]
        else:
             # Use a fixed portion of the history as input
             fixed_history_length = int(np.floor(self.sequence_length * self.history_fraction))
             motion_input = full_trajectory[:, :fixed_history_length, :] # Shape: [B, fixed_hist, 3]
        # --------------------------
        
        # 1. Embed past motion
        motion_feats = self.motion_linear(motion_input)  # [B, input_len, motion_hidden_dim]

        # 2. Encode point cloud (using PointNet2SemSegSSGShape)
        point_cloud_6d = point_cloud.repeat(1, 1, 2)  # [B, N, 3] -> [B, N, 6]
        scene_feats, scene_global_feats = self.scene_encoder(point_cloud_6d)  # [B, F, N_points], [B, F]

        # 3. Process motion with motion encoder
        # The input to motion_encoder might need padding if we used dynamic history here.
        # Using fixed history simplifies this.
        motion_embedding = self.motion_encoder(motion_feats)
        # [B, sequence_length, motion_latent_dim]

        # 4. Combine global scene features with motion embedding for cross-modal fusion
        out_seq_len = motion_embedding.shape[1]
        
        # Expand scene_global_feats to match sequence length
        cross_modal_embedding = scene_global_feats.unsqueeze(1).repeat(1, out_seq_len, 1)
        
        # Concatenate scene global features and motion embedding
        cross_modal_embedding = torch.cat([cross_modal_embedding, motion_embedding], dim=2)
        # [B, sequence_length, scene_feats_dim + motion_latent_dim]

        # 5. Apply embedding layer (like in crossmodal_net.py)
        cross_modal_embedding = self.embedding_layer(cross_modal_embedding)
        
        # 6. Final encoding with output encoder
        cross_modal_embedding = self.output_encoder(cross_modal_embedding)
        # [B, sequence_length, cross_hidden_dim]

        # 7. Predict trajectory for all tokens in the sequence
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
        Losses are averaged per sequence based on valid points before batch averaging.
        
        When config.use_first_frame_only is True:
            - `predictions` tensor has shape [B, sequence_length-1, 3]
            - Loss is calculated only on the future frames (GT indices 1 to N).
            - Reconstruction loss is set to 0.

        Args:
            predictions: The model's output tensor. Shape depends on config.use_first_frame_only.
                         [B, sequence_length, 3] if False.
                         [B, sequence_length-1, 3] if True.
            batch: The batch dictionary from the DataLoader, containing ground truth.
                   Expected keys: 'full_positions' [B, sequence_length, 3]
                                  'full_attention_mask' [B, sequence_length]

        Returns:
            torch.Tensor: The computed total loss value (scalar).
            dict: A dictionary containing individual loss components (batch averaged).
        """
        gt_full_positions = batch['full_positions'].to(predictions.device)
        gt_full_mask = batch['full_attention_mask'].to(predictions.device) # Shape [B, seq_len]
        batch_size = gt_full_positions.shape[0]
        device = predictions.device

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
            dynamic_future_mask_for_loss = dynamic_future_mask[:, 1:] # [B, seq_len-1]
            # Predictions tensor already corresponds to GT indices 1 onwards
            pred_future_positions = predictions # [B, seq_len-1, 3]
        else:
            # Use full GT and prediction for standard mode loss calculation
            gt_future_positions = gt_full_positions
            pred_future_positions = predictions
            dynamic_future_mask_for_loss = dynamic_future_mask
            
        # --- Calculate L1 Reconstruction Loss (XYZ History) --- 
        # This is only calculated if not use_first_frame_only
        if not self.config.use_first_frame_only:
            loss_rec_per_coord = F.l1_loss(predictions, gt_full_positions, reduction='none') # Compare full sequences
            loss_rec_per_point = torch.sum(loss_rec_per_coord, dim=-1) # [B, seq_len]
            masked_loss_rec = loss_rec_per_point * dynamic_history_mask # Apply DYNAMIC history mask [B, seq_len]
            
            sum_loss_rec_per_seq = torch.sum(masked_loss_rec, dim=1) # [B]
            num_valid_hist_per_seq = torch.sum(dynamic_history_mask, dim=1) # [B]
            valid_rec_mask = num_valid_hist_per_seq > 0
            mean_loss_rec_per_seq = torch.zeros_like(sum_loss_rec_per_seq)
            mean_loss_rec_per_seq[valid_rec_mask] = sum_loss_rec_per_seq[valid_rec_mask] / num_valid_hist_per_seq[valid_rec_mask]
            mean_rec_loss_xyz = torch.sum(mean_loss_rec_per_seq) / torch.sum(valid_rec_mask) if torch.sum(valid_rec_mask) > 0 else torch.tensor(0.0, device=device)
            rec_loss_weight = self.config.lambda_rec_xyz
        else:
            mean_rec_loss_xyz = torch.tensor(0.0, device=device)
            rec_loss_weight = 0.0
        # ------------------------------------------------------

        # --- Calculate L1 Path Loss (XYZ Future) --- 
        # Compare predicted future with GT future using the future mask
        loss_path_per_coord = F.l1_loss(pred_future_positions, gt_future_positions, reduction='none') # [B, relevant_len, 3]
        loss_path_per_point = torch.sum(loss_path_per_coord, dim=-1) # [B, relevant_len]
        
        # Ensure the mask matches the dimensions being compared
        masked_loss_path = loss_path_per_point * dynamic_future_mask_for_loss # [B, relevant_len]
        
        sum_loss_path_per_seq = torch.sum(masked_loss_path, dim=1) # [B]
        num_valid_future_per_seq = torch.sum(dynamic_future_mask_for_loss, dim=1) # [B]
        valid_path_mask = num_valid_future_per_seq > 0
        mean_loss_path_per_seq = torch.zeros_like(sum_loss_path_per_seq)
        mean_loss_path_per_seq[valid_path_mask] = sum_loss_path_per_seq[valid_path_mask] / num_valid_future_per_seq[valid_path_mask]
        mean_path_loss_xyz = torch.sum(mean_loss_path_per_seq) / torch.sum(valid_path_mask) if torch.sum(valid_path_mask) > 0 else torch.tensor(0.0, device=device)
        # -------------------------------------------

        # --- Calculate L1 Destination Loss (XYZ) --- 
        # Find the index of the last valid point in the *original* GT sequence
        last_valid_indices_gt = actual_lengths.long() - 1
        last_valid_indices_gt = torch.clamp(last_valid_indices_gt, min=0)
        
        batch_indices = torch.arange(batch_size, device=device)
        
        # Get the last valid ground truth position
        last_gt_positions = gt_full_positions[batch_indices, last_valid_indices_gt, :]
        
        # Get the corresponding predicted position
        if self.config.use_first_frame_only:
            # Prediction indices are shifted by 1 relative to GT indices
            # Last valid prediction index corresponds to GT index last_valid_indices_gt - 1
            last_valid_indices_pred = last_valid_indices_gt - 1
            # Clamp to ensure index is valid for prediction tensor (length seq_len-1)
            last_valid_indices_pred = torch.clamp(last_valid_indices_pred, min=0, max=predictions.shape[1] - 1) 
            last_pred_positions = predictions[batch_indices, last_valid_indices_pred, :]
        else:
            # Standard case: index directly corresponds
            last_pred_positions = predictions[batch_indices, last_valid_indices_gt, :]
            
        # Calculate L1 loss per destination point
        dest_loss_per_seq = torch.sum(F.l1_loss(last_pred_positions, last_gt_positions, reduction='none'), dim=1) # [B]
        
        # Only consider sequences that have at least two points (actual_lengths >= 2 for future prediction)
        # or at least one point in the standard case.
        min_len_for_dest = 2 if self.config.use_first_frame_only else 1
        valid_dest_mask = actual_lengths >= min_len_for_dest
        masked_dest_loss = dest_loss_per_seq * valid_dest_mask.float()
        
        # Batch average of destination loss (only over valid sequences)
        mean_dest_loss_xyz = torch.sum(masked_dest_loss) / torch.sum(valid_dest_mask) if torch.sum(valid_dest_mask) > 0 else torch.tensor(0.0, device=device)
        # -----------------------------------------

        # --- Combine Losses --- 
        # rec_loss_weight is determined above based on use_first_frame_only
        total_loss = (rec_loss_weight * mean_rec_loss_xyz + 
                      self.config.lambda_path_xyz * mean_path_loss_xyz + 
                      self.config.lambda_dest_xyz * mean_dest_loss_xyz)
                     
        # Store individual batch-averaged losses for logging/debugging
        loss_dict = {
            'total_loss': total_loss,
            'rec_loss_xyz': rec_loss_weight * mean_rec_loss_xyz,
            'path_loss_xyz': self.config.lambda_path_xyz * mean_path_loss_xyz,
            'dest_loss_xyz': self.config.lambda_dest_xyz * mean_dest_loss_xyz
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
    
    # Create dummy input data
    past_traj_dummy = torch.randn(config.batch_size, config.history_length, config.object_motion_dim)
    point_cloud_dummy = torch.randn(config.batch_size, config.sample_points, config.point_cloud_dim)
    
    # Instantiate model
    model = GIMO_ADT_Model(config)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Forward pass
    output = model(past_traj_dummy, point_cloud_dummy)
    
    print(f"Input past trajectory shape: {past_traj_dummy.shape}")
    print(f"Input point cloud shape: {point_cloud_dummy.shape}")
    print(f"Output predicted future shape: {output.shape}")
    
    # Check output shape
    expected_shape = (config.batch_size, config.sequence_length, config.object_motion_dim)
    assert output.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, Got {output.shape}"
    print("Output shape check passed.") 