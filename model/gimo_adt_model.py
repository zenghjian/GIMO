import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os # Added for path manipulation
import matplotlib.pyplot as plt # Added for visualization

# Actual GIMO components
from .pointnet_plus2 import PointNet2SemSegSSGShape, PointNet, MyFPModule
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
        
        # Check if bounding box processing is disabled
        self.use_bbox = not getattr(config, 'no_bbox', False)
        
        # Still keep fixed trajectory_length for model definition
        self.sequence_length = config.trajectory_length 

        # --- Scene/Motion Encoding Components ---

        # 1. Motion Embedding
        self.motion_linear = nn.Linear(config.object_motion_dim, config.motion_hidden_dim)

        # 2. Point Cloud Encoder
        self.scene_encoder = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})

        # Bounding Box related layers (as per diagram)
        self.fp_layer = MyFPModule()
        self.bbox_pointnet = PointNet(config.scene_feats_dim) # Use existing scene_feats_dim
        
        # Encoder for fusing motion and bbox_scene_features
        # Adjust input dimension based on whether no_bbox is enabled
        motion_bbox_encoder_input_dim = config.motion_hidden_dim
        if not self.use_bbox:
            # If bbox disabled, we only use motion features
            self.motion_bbox_encoder = PerceiveEncoder(
                n_input_channels=config.motion_hidden_dim,
                n_latent=self.sequence_length,
                n_latent_channels=config.motion_latent_dim,
                n_self_att_heads=config.motion_n_heads,    
                n_self_att_layers=config.motion_n_layers,   
                dropout=config.dropout
            )
        else:
            # With bbox enabled, we use both motion and bbox features
            self.motion_bbox_encoder = PerceiveEncoder(
                n_input_channels=config.motion_hidden_dim + config.scene_feats_dim,
                n_latent=self.sequence_length,
                n_latent_channels=config.motion_latent_dim,
                n_self_att_heads=config.motion_n_heads,    
                n_self_att_layers=config.motion_n_layers,   
                dropout=config.dropout
            )

        # --- Category Embedding (simplified approach) ---
        if not config.no_text_embedding:
            self.category_embedding = nn.Embedding(
                num_embeddings=config.num_object_categories,
                embedding_dim=config.category_embed_dim
            )
        
        # Final Embedded Layer (second fusion)
        # Prepare input dims for embedding layer
        # This layer will now potentially receive category embeddings as well
        embed_dim = config.motion_latent_dim + config.scene_feats_dim
        if not config.no_text_embedding:
            embed_dim += config.category_embed_dim
        
        self.embedding_layer = PositionwiseFeedForward(
            d_in=embed_dim, 
            d_hid=embed_dim, 
            dropout=config.dropout
        )
        
        
        # --- Output components ---
        self.output_encoder = PerceiveEncoder(
            n_input_channels=embed_dim,
            n_latent=self.sequence_length,  # Required parameter for number of latent tokens
            n_latent_channels=config.output_latent_dim,
            n_self_att_heads=config.output_n_heads,
            n_self_att_layers=config.output_n_layers,
            dropout=config.dropout
        )
        
        # Final output layer (predicts full motion trajectory)
        self.outputlayer = nn.Linear(config.output_latent_dim, config.object_motion_dim)
    
    def forward(self, input_trajectory, point_cloud, bounding_box_corners=None, object_category_ids=None):
        """
        Forward pass of the GIMO ADT model with 3D bounding box integration.
        
        Args:
            input_trajectory: [batch_size, input_length, 9] - history trajectory (positions and 6D rotations)
            point_cloud: [batch_size, num_points, 3] - scene point cloud
            bounding_box_corners: [batch_size, input_length, 8, 3] - 3D bbox corners for each input timestep (optional if no_bbox=True)
            object_category_ids: [batch_size] - category IDs for objects (optional if no_text_embedding=True)
            
        Returns:
            torch.Tensor: Predicted full trajectory [batch_size, sequence_length, 9]
        """
        batch_size = input_trajectory.shape[0]
        input_length = input_trajectory.shape[1]  # length of input trajectory
        
        # --- Check if object_category_ids is provided when needed ---
        if not self.config.no_text_embedding and object_category_ids is None:
            raise ValueError("object_category_ids is required when no_text_embedding=False")
            
        # --- Check if bounding_box_corners is provided when needed ---
        if self.use_bbox and bounding_box_corners is None:
            raise ValueError("bounding_box_corners is required when no_bbox=False")
            
        # Process Input Trajectory
        # Input trajectory to motion features [B, T, motion_hidden_dim]
        f_m = self.motion_linear(input_trajectory)
        
        # Process Point Cloud
        # Process the scene point cloud to get per-point features and global features
        # PointNet2SemSegSSGShape expects point_cloud to be [B, N, 6] where the last 3 dimensions are duplicated
        # Convert [B, N, 3] to [B, N, 6] by duplicating the XYZ values
        point_cloud_6d = torch.cat([point_cloud, point_cloud], dim=2)  # [B, N, 6]
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            scene_feats_per_point, scene_global_feats = self.scene_encoder(point_cloud_6d)
        
        # Process Bounding Box (if enabled)
        if self.use_bbox and bounding_box_corners is not None:
            num_bbox_corners = bounding_box_corners.shape[2]  # should be 8
            
            # First reshape everything for feature propagation to bbox corners
            point_cloud_repeated_for_fp = point_cloud.unsqueeze(1).repeat(1, input_length, 1, 1) # [B, T, N, 3]
            point_cloud_for_fp = point_cloud_repeated_for_fp.reshape(batch_size * input_length, -1, 3).contiguous() # [B*T, N, 3]

            scene_feats_per_point_for_fp = scene_feats_per_point.unsqueeze(2).repeat(1, 1, input_length, 1) # [B, F, T, N]
            scene_feats_per_point_for_fp = scene_feats_per_point_for_fp.permute(0, 2, 1, 3).reshape(batch_size * input_length, self.config.scene_feats_dim, -1).contiguous() # [B*T, F, N]
            
            bbox_corners_for_fp = bounding_box_corners.reshape(batch_size * input_length, num_bbox_corners, 3).contiguous() # [B*T, 8, 3]
            
            # Propagate features to bounding box corners
            # self.fp_layer expects unknown [B*T, 8, 3], known [B*T, N, 3], known_feats [B*T, F, N]
            # Output: propagated_feats [B*T, F, 8]
            propagated_feats_to_bbox = self.fp_layer(
                unknown=bbox_corners_for_fp,  # target_points [B*T, 8, 3] 
                known=point_cloud_for_fp,     # source_points [B*T, N, 3]
                known_feats=scene_feats_per_point_for_fp  # source_features [B*T, F, N]
            )
            
            # Process propagated features with PointNet for each bounding box
            # self.bbox_pointnet expects [B*T, F, 8]
            # Output: f_s_b_per_ts [B*T, scene_feats_dim]
            f_s_b_per_ts = self.bbox_pointnet(propagated_feats_to_bbox) 

            # Reshape f_s_b to match f_m sequence length
            # Output: f_s_b [B, T, scene_feats_dim]
            f_s_b = f_s_b_per_ts.reshape(batch_size, input_length, self.config.scene_feats_dim)
            
            # f_m is [B, T, motion_hidden_dim]
            # f_s_b is [B, T, scene_feats_dim]
            fused_motion_bbox_input = torch.cat([f_m, f_s_b], dim=2) # [B, T, motion_hidden_dim + scene_feats_dim]
        else:
            # If not using bbox, just use motion features directly
            fused_motion_bbox_input = f_m  # [B, T, motion_hidden_dim]
        
        # encoded_motion_bbox will have shape [B, self.sequence_length, motion_latent_dim]
        encoded_motion_bbox = self.motion_bbox_encoder(fused_motion_bbox_input)

        # --- Generate Category Embeddings (if used) ---
        category_embeddings_expanded = None
        if not self.config.no_text_embedding and object_category_ids is not None:
            category_embeddings = self.category_embedding(object_category_ids) # [B, category_embed_dim]
            category_embeddings_expanded = category_embeddings.unsqueeze(1).repeat(1, self.sequence_length, 1) # [B, sequence_length, category_embed_dim]

        # scene_global_feats is [B, scene_feats_dim]
        # encoded_motion_bbox is [B, sequence_length, motion_latent_dim]
        scene_global_feats_expanded = scene_global_feats.unsqueeze(1).repeat(1, self.sequence_length, 1) # [B, sequence_length, scene_feats_dim]
        
        features_to_fuse = [
            scene_global_feats_expanded,  # Global scene context
            encoded_motion_bbox,          # Motion + bounding box features
        ]

        if category_embeddings_expanded is not None:
            features_to_fuse.append(category_embeddings_expanded)
        
        # Concatenate all features along the last dimension
        # [B, sequence_length, scene_feats_dim + motion_latent_dim + category_embed_dim]
        final_fused_input = torch.cat(features_to_fuse, dim=2)


        cross_modal_embedding = self.embedding_layer(final_fused_input) # Output: [B, sequence_length, embed_input_dim]

        
        # Process with output encoder
        # output_features is [B, sequence_length, output_latent_dim]
        cross_modal_embedding = self.output_encoder(cross_modal_embedding)
        all_predictions = self.outputlayer(cross_modal_embedding)

        return all_predictions

    def compute_loss(self, predictions, batch, epoch=None, batch_idx=None, vis_save_dir=None, sample_name_for_vis=None):
        """
        Computes the loss for the predicted trajectories using L1 norm,
        incorporating dynamic history/future split based on attention mask.
        Losses are simplified to translation (position) and orientation components.

        Now predictions tensor always has shape [B, sequence_length, motion_dim]
        regardless of config.use_first_frame_only setting.

        Args:
            predictions: The model's output tensor with shape [B, sequence_length, motion_dim].
            batch: The batch dictionary from the DataLoader, containing ground truth.
                   Expected keys: 'full_poses' [B, sequence_length, motion_dim]
                                  'full_attention_mask' [B, sequence_length]
            epoch (optional): Current epoch number, for visualization.
            batch_idx (optional): Current batch index, for visualization.
            vis_save_dir (optional): Directory to save visualizations.
            sample_name_for_vis (optional): Sample name for visualization filename.


        Returns:
            torch.Tensor: The computed total loss value (scalar).
            dict: A dictionary containing individual loss components (batch averaged).
        """
        gt_full_poses = batch['full_poses'].to(predictions.device)
        gt_full_mask = batch['full_attention_mask'].to(predictions.device) # Shape [B, seq_len]
        batch_size = gt_full_poses.shape[0]
        device = predictions.device

        # Split poses into positions and rotations for separate loss calculation
        position_dim = 3  # Assuming first 3 dimensions are x, y, z
        rotation_dim = 6  # 6D rotation representation

        # Extract ground truth positions and rotations
        gt_full_positions = gt_full_poses[..., :position_dim]
        gt_full_rotations = gt_full_poses[..., position_dim:position_dim+rotation_dim]

        # Extract predicted positions and rotations
        pred_positions = predictions[..., :position_dim]
        pred_rotations = predictions[..., position_dim:position_dim+rotation_dim]

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

        # For loss calculation, use complete sequences since predictions now always includes all frames
        gt_future_positions = gt_full_positions
        gt_future_rotations = gt_full_rotations
        pred_future_positions = pred_positions
        pred_future_rotations = pred_rotations
        
        # For use_first_frame_only mode, we still only want to compute future loss on frames after the first
        if self.config.use_first_frame_only:
            # Create a mask that only includes future frames (excluding the first frame)
            # This is basically dynamic_future_mask but we know history length is 1
            future_only_mask = (indices >= 1).float() * gt_full_mask # [B, seq_len]
            dynamic_future_mask_for_loss = future_only_mask
        else:
            # Standard mode, use all frames for loss but with history/future split
            dynamic_future_mask_for_loss = dynamic_future_mask
        
        # --- Calculate Translation Loss (Position) ---
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
        loss_ori_per_coord = F.l1_loss(pred_future_rotations, gt_future_rotations, reduction='none') # [B, relevant_len, 6]
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
            hist_gt_poses = gt_full_poses  # [B, seq_len, 9]
            hist_pred_poses = predictions  # [B, seq_len, 9]
            
            # Calculate reconstruction loss for full poses (positions + rotations)
            loss_rec_per_coord = F.l1_loss(hist_pred_poses, hist_gt_poses, reduction='none')  # [B, seq_len, 9]
            loss_rec_per_point = torch.sum(loss_rec_per_coord, dim=-1)  # [B, seq_len]
            masked_loss_rec = loss_rec_per_point * hist_mask  # Apply mask [B, seq_len]
            
            sum_loss_rec_per_seq = torch.sum(masked_loss_rec, dim=1)  # [B]
            num_valid_hist_points_per_seq = torch.sum(hist_mask, dim=1)  # [B]
            valid_hist_mask = num_valid_hist_points_per_seq > 0
            mean_loss_rec_per_seq = torch.zeros_like(sum_loss_rec_per_seq)
            mean_loss_rec_per_seq[valid_hist_mask] = sum_loss_rec_per_seq[valid_hist_mask] / num_valid_hist_points_per_seq[valid_hist_mask]
            mean_rec_loss = torch.sum(mean_loss_rec_per_seq) / torch.sum(valid_hist_mask) if torch.sum(valid_hist_mask) > 0 else torch.tensor(0.0, device=device)
        elif self.config.use_first_frame_only:
            # For first_frame_only mode, calculate reconstruction loss just for the first frame
            # Now predictions includes the first frame (predictions[:, 0, :])
            first_frame_gt = gt_full_poses[:, 0, :]  # [B, 9]
            first_frame_pred = predictions[:, 0, :]  # [B, 9]
            
            # Create a mask for valid first frames
            first_frame_mask = gt_full_mask[:, 0]  # [B]
            
            # Calculate reconstruction loss for first frame
            first_frame_loss = F.l1_loss(first_frame_pred, first_frame_gt, reduction='none')  # [B, 9]
            first_frame_loss_per_batch = torch.sum(first_frame_loss, dim=1)  # [B]
            
            # Apply mask and average
            valid_first_frames = torch.sum(first_frame_mask)
            if valid_first_frames > 0:
                masked_first_frame_loss = first_frame_loss_per_batch * first_frame_mask
                mean_rec_loss = torch.sum(masked_first_frame_loss) / valid_first_frames
            else:
                mean_rec_loss = torch.tensor(0.0, device=device)
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
        last_gt_poses = gt_full_poses[batch_indices, last_valid_indices]  # [B, 9]
        
        # Predictions now always include the full sequence with same indexing as GT
        last_pred_poses = predictions[batch_indices, last_valid_indices]  # [B, 9]
        
        # Calculate destination loss for both positions and rotations
        dest_loss_per_coord = F.l1_loss(last_pred_poses, last_gt_poses, reduction='none')  # [B, 9]
        
        # Split into position and rotation components
        dest_trans_loss_per_seq = torch.sum(dest_loss_per_coord[:, :position_dim], dim=1)  # [B]
        dest_ori_loss_per_seq = torch.sum(dest_loss_per_coord[:, position_dim:position_dim+rotation_dim], dim=1)  # [B]
        
        # Apply mask to handle invalid sequences
        valid_dest_mask = last_valid_indices >= 0 # Check if index is valid (>=0)
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

        # --- Visualization Logic ---
        if vis_save_dir and sample_name_for_vis and epoch is not None and batch_idx is not None:
            try:
                # Ensure only visualizing for the first item in the batch (if batch_size > 1, though for overfitting it's 1)
                if batch_size > 0:
                    os.makedirs(vis_save_dir, exist_ok=True)
                    
                    # Prepare data for visualization (first sample in batch)
                    gt_poses_vis = gt_full_poses[0, :, :position_dim].norm(dim=-1).cpu().numpy() # L2 norm of position
                    gt_mask_vis = gt_full_mask[0].cpu().numpy()
                    hist_mask_vis = dynamic_history_mask[0].cpu().numpy()
                    future_mask_vis = dynamic_future_mask[0].cpu().numpy()
                    hist_len_vis = dynamic_history_lengths[0].item()

                    timesteps = np.arange(self.sequence_length)

                    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                    fig.suptitle(f'Mask Visualization - {sample_name_for_vis}\\nEpoch {epoch}, Batch {batch_idx}, Hist Len: {hist_len_vis}', fontsize=10)

                    # Plot GT Full Pose (L2 norm of position)
                    axs[0].bar(timesteps, gt_poses_vis, color='blue', alpha=0.7)
                    axs[0].set_title('GT Full Pose (Position L2 Norm)', fontsize=8)
                    axs[0].set_ylabel('L2 Norm', fontsize=8)
                    
                    # Plot GT Full Mask
                    axs[1].bar(timesteps, gt_mask_vis, color='green', alpha=0.7)
                    axs[1].set_title('GT Full Mask', fontsize=8)
                    axs[1].set_ylabel('Mask Value', fontsize=8)
                    axs[1].set_yticks([0, 1])

                    # Plot Dynamic History Mask
                    axs[2].bar(timesteps, hist_mask_vis, color='red', alpha=0.7)
                    axs[2].set_title(f'Dynamic History Mask (Length: {hist_len_vis})', fontsize=8)
                    axs[2].set_ylabel('Mask Value', fontsize=8)
                    axs[2].set_yticks([0, 1])

                    # Plot Dynamic Future Mask
                    axs[3].bar(timesteps, future_mask_vis, color='purple', alpha=0.7)
                    axs[3].set_title('Dynamic Future Mask', fontsize=8)
                    axs[3].set_ylabel('Mask Value', fontsize=8)
                    axs[3].set_yticks([0, 1])
                    
                    plt.xlabel('Timestep', fontsize=8)
                    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
                    
                    save_filename = f"{sample_name_for_vis}_epoch{epoch}_batch{batch_idx}_model_masks.png"
                    save_path = os.path.join(vis_save_dir, save_filename)
                    plt.savefig(save_path)
                    plt.close(fig)
                    # print(f"Saved mask visualization to {save_path}") # Optional: for debugging
            except Exception as e:
                print(f"Error during mask visualization in GIMO_ADT_Model: {e}")
                # import traceback
                # traceback.print_exc() # For more detailed error logging if needed

        return total_loss, loss_dict 