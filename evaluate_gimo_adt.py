#!/usr/bin/env python3
# Evaluation script for GIMO ADT Motion Prediction Model

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from torch.utils.data import DataLoader
import time
from functools import partial # Import partial for binding args to collate_fn
from torch.utils.data.dataloader import default_collate # Import default_collate

# --- Imports from our project ---
from model.gimo_adt_model import GIMO_ADT_Model
from dataset.gimo_multi_sequence_dataset import GIMOMultiSequenceDataset
from dataset.gimo_adt_trajectory_dataset import GimoAriaDigitalTwinTrajectoryDataset # Needed for denormalization access potentially
from config.adt_config import ADTObjectMotionConfig
from train_adt import gimo_collate_fn # Import the custom collate function
from utils.visualization import visualize_trajectory, visualize_prediction, visualize_full_trajectory # Import visualization utils

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GIMO ADT trajectory prediction model")
    
    # --- Model and Checkpoint ---
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained GIMO_ADT_Model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_gimo_adt",
        help="Directory to save evaluation results"
    )
    
    # --- Dataset Options (using config from checkpoint primarily) ---
    # These might override config if needed, but generally config is preferred
    parser.add_argument(
        "--adt_dataroot",
        type=str,
        default=None, # Default to None, use config from checkpoint
        help="Override path to ADT dataroot (directory containing sequences/split files)"
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default=None, # Default to None, use config from checkpoint
        help="Override path to test split file"
    )
    
    # --- Evaluation Options ---
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16, # Can override config batch size for evaluation
        help="Batch size for evaluation"
    )
    # parser.add_argument(
    #     "--num-samples", # GIMO is deterministic, so only 1 sample needed
    #     type=int,
    #     default=1,
    #     help="Number of samples to generate (GIMO is deterministic, so usually 1)"
    # )
    
    # --- Visualization Options ---
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of trajectories"
    )
    parser.add_argument(
        "--num_vis_samples",
        type=int,
        default=50, # Number of trajectory samples to visualize
        help="Number of trajectory samples to visualize"
    )
    parser.add_argument(
        "--visualize_bbox",
        action="store_true",
        help="Enable visualization of bounding boxes"
    )
    
    # --- Other Options ---
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4, # Can override config workers for evaluation
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--global_cache_dir",
        type=str,
        default=None,
        help="Path to a shared global directory for trajectory cache (overrides cache within output_dir)"
    )
    
    # --- Model Configuration Overrides ---
    parser.add_argument(
        "--use_first_frame_only",
        action="store_true",
        help="Override config: Use only the first frame as input to predict the future"
    )
    parser.add_argument(
        "--show_ori_arrows",
        action="store_true",
        help="Show orientation arrows in visualizations"
    )

    return parser.parse_args()

def load_model_and_config(checkpoint_path, device):
    """
    Load GIMO_ADT_Model and its config from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pth)
        device: Device to load the model on
        
    Returns:
        model: Loaded GIMO_ADT_Model
        config: Configuration object from the checkpoint
        best_epoch (int): The epoch number at which this checkpoint was saved (usually the best validation epoch). Returns 0 if not found.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'config'. Cannot instantiate model.")
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")
        
    # Load config (convert dict back to Namespace or use directly if already object)
    config_dict = checkpoint['config']
    config = argparse.Namespace(**config_dict) # Convert dict back to Namespace
    
    print("Configuration loaded from checkpoint:")
    print(config)

    # Create model instance using loaded config
    model = GIMO_ADT_Model(config).to(device)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode
    
    param_count = sum(p.numel() for p in model.parameters())
    param_count_millions = param_count / 1_000_000
    print(f"Loaded GIMO_ADT_Model with {param_count_millions:.2f}M parameters.")
    
    # Get the epoch number
    best_epoch = checkpoint.get('epoch', 0) # Default to 0 if epoch key doesn't exist
    if best_epoch > 0:
        print(f"Checkpoint was saved at epoch: {best_epoch}")
    else:
        print("Warning: Epoch number not found in checkpoint.")
        
    return model, config, best_epoch

def denormalize_trajectory(normalized_trajectory, normalization_params):
    """Denormalize a trajectory using provided parameters."""
    if not normalization_params['is_normalized']:
        return normalized_trajectory # Return as is if not normalized

    # Ensure parameters are tensors on the correct device
    # Make sure keys exist before trying to access and move to device
    scene_min_val = normalization_params.get('scene_min')
    scene_max_val = normalization_params.get('scene_max')
    scene_scale_val = normalization_params.get('scene_scale')

    if not all(isinstance(p, torch.Tensor) for p in [scene_min_val, scene_max_val, scene_scale_val]):
         print("Warning: Invalid normalization parameters found during denormalization. Returning original trajectory.")
         return normalized_trajectory

    scene_min = scene_min_val.to(normalized_trajectory.device)
    scene_max = scene_max_val.to(normalized_trajectory.device) # Not strictly needed for denorm, but good practice
    scene_scale = scene_scale_val.to(normalized_trajectory.device)

    # Denormalize: [-1, 1] -> [0, 1] -> world
    denormalized = (normalized_trajectory + 1.0) / 2.0
    denormalized = denormalized * scene_scale + scene_min
    return denormalized

def compute_metrics_for_sample(pred_future, gt_future, future_mask):
    """
    Compute L1 Mean, L2 Mean (for RMSE), and FDE for a single trajectory sample.

    Args:
        pred_future (torch.Tensor): Predicted future trajectory [Fut_Len, 3] (MUST be denormalized)
        gt_future (torch.Tensor): Ground truth future trajectory [Fut_Len, 3] (MUST be denormalized)
        future_mask (torch.Tensor): Mask for valid future points [Fut_Len]

    Returns:
        l1_mean (torch.Tensor): Mean L1 distance (scalar)
        rmse_ade (torch.Tensor): Mean L2 distance (RMSE analog for ADE) (scalar)
        fde (torch.Tensor): Final Displacement Error (scalar)
    """
    # Ensure tensors are on the same device
    device = pred_future.device
    gt_future = gt_future.to(device)
    future_mask = future_mask.to(device)

    # Calculate L1 distance for all points
    l1_diff = torch.abs(pred_future - gt_future) # Shape: [Fut_Len, 3]
    # Calculate per-timestep L2 distance (Euclidean norm)
    l2_dist_per_step = torch.norm(pred_future - gt_future, dim=-1) # Shape: [Fut_Len]

    # Expand mask to match 3D coordinates (for L1)
    future_mask_expanded = future_mask.unsqueeze(-1).expand_as(l1_diff) # Shape: [Fut_Len, 3]

    # Count valid points (use original 1D mask)
    num_valid_points = future_mask.sum()
    if num_valid_points == 0:
        # Return zeros or NaNs if no valid points to avoid division by zero
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device) # l1_mean, rmse_ade, fde
        
    num_valid_coords = num_valid_points * 3 # Total number of valid coordinate values

    # --- L1 Mean (MAE) Calculation ---
    # Mask out invalid points
    masked_l1_diff = l1_diff * future_mask_expanded
    # Sum distances over all valid coordinates
    sum_l1_diff = masked_l1_diff.sum()
    # Calculate average L1 distance
    l1_mean = sum_l1_diff / num_valid_coords

    # --- RMSE / ADE (L2) Calculation ---
    # Mask out invalid points
    masked_l2_dist = l2_dist_per_step * future_mask # Use 1D mask here
    # Sum L2 distances over all valid timesteps
    sum_l2_dist = masked_l2_dist.sum()
    # Calculate average L2 distance over valid timesteps
    rmse_ade = sum_l2_dist / num_valid_points

    # --- FDE Calculation ---
    # Find the index of the last valid point (using 1D mask)
    last_valid_index = num_valid_points.long() - 1
    last_valid_index = torch.clamp(last_valid_index, min=0) # Clamp index

    # Get the predicted and ground truth points at the final valid timestep
    final_pred_point = pred_future[last_valid_index]
    final_gt_point = gt_future[last_valid_index]

    # Calculate FDE as the L2 distance between these final points
    fde = torch.norm(final_pred_point - final_gt_point, dim=-1)

    return l1_mean, rmse_ade, fde

def evaluate(model, config, args, best_epoch):
    """
    Run the evaluation loop.
    
    Args:
        model: The loaded GIMO_ADT_Model.
        config: The configuration object (usually loaded from checkpoint).
        args: Command line arguments for evaluation.
        best_epoch (int): The epoch number the model checkpoint was saved at.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # --- Setup Dataset and DataLoader ---
    print("Setting up test dataset...")
    # Use config values from checkpoint, potentially overridden by args
    test_sequences = None
    checkpoint_dir = os.path.dirname(args.model_path)
    val_split_path = os.path.join(checkpoint_dir, 'val_sequences.txt')
    dataroot_override = args.adt_dataroot # Store override if provided
    
    if os.path.exists(val_split_path):
        try:
            print(f"Loading test sequences from saved validation split: {val_split_path}")
            with open(val_split_path, 'r') as f:
                test_sequences = [line.strip() for line in f if line.strip()]
            if not test_sequences:
                 print(f"Warning: {val_split_path} is empty.")
                 test_sequences = None # Force fallback
            else:
                 print(f"Loaded {len(test_sequences)} test sequences.")
                 # Optional: Adjust paths if dataroot_override is given and paths are relative
                 if dataroot_override:
                      adjusted_sequences = []
                      for seq in test_sequences:
                          # Check if path is relative and needs joining with dataroot
                          if not os.path.isabs(seq) and not seq.startswith(dataroot_override):
                              base_name = os.path.basename(seq)
                              adjusted_path = os.path.join(dataroot_override, base_name)
                              if os.path.exists(adjusted_path):
                                 adjusted_sequences.append(adjusted_path)
                              else:
                                 print(f"Warning: Could not adjust path for {seq} using override {dataroot_override}")
                                 adjusted_sequences.append(seq) # Keep original if adjustment fails
                          else:
                              adjusted_sequences.append(seq)
                      test_sequences = adjusted_sequences
                      print(f"Adjusted sequence paths using dataroot override: {dataroot_override}")
                             
        except Exception as e:
            print(f"Warning: Error loading {val_split_path}: {e}. Will attempt fallback.")
            test_sequences = None
    else:
        print(f"Warning: val_sequences.txt not found in {checkpoint_dir}. Attempting fallback using config.")

    # Fallback logic if val_sequences.txt was not loaded successfully
    if test_sequences is None:
        print("Attempting fallback: Using adt_dataroot from config/args...")
        dataroot = dataroot_override if dataroot_override is not None else config.adt_dataroot
        if os.path.isdir(dataroot):
             # Use all sequences in dataroot as test set (assuming no split info available)
             print(f"Warning: Using ALL sequences found in {dataroot} as test set due to missing split info.")
             # Import sequence utils dynamically if needed for fallback
             try:
                 from ariaworldgaussians.adt_sequence_utils import find_adt_sequences
                 test_sequences = find_adt_sequences(dataroot)
             except ImportError:
                 print("Warning: Cannot import find_adt_sequences. Manually scanning directory.")
                 test_sequences = [os.path.join(dataroot, item) for item in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, item))]
                 
             if not test_sequences:
                  print(f"Error: No sequences found in fallback dataroot {dataroot}.")
                  return None
             print(f"Found {len(test_sequences)} sequences in fallback dataroot.")
        elif os.path.exists(dataroot): # Check if dataroot itself is a single sequence path
             test_sequences = [dataroot]
             print(f"Using single sequence {dataroot} as test set (fallback).")
        else:
             print(f"Error: Fallback failed. Dataroot path {dataroot} not found or invalid.")
             return None

    # Use cache dir based on output dir to avoid conflicts with training cache
    # Prioritize global_cache_dir if provided
    if args.global_cache_dir:
        eval_cache_dir = args.global_cache_dir
        os.makedirs(eval_cache_dir, exist_ok=True)
        print(f"Using global cache directory for evaluation: {eval_cache_dir}")
    else:
        eval_cache_dir = os.path.join(args.output_dir, 'trajectory_cache_eval')
        os.makedirs(eval_cache_dir, exist_ok=True)
        print(f"Using evaluation-specific cache directory: {eval_cache_dir}")

    test_dataset = GIMOMultiSequenceDataset(
        sequence_paths=test_sequences,
        config=config, # Use loaded config
        cache_dir=eval_cache_dir, # Use separate cache for eval
        use_cache=True # Enable caching for evaluation
    )

    if len(test_dataset) == 0:
        print("Error: Test dataset is empty. Check test split file and sequence paths.")
        return None

    # Use the collate function, binding the test dataset instance
    eval_collate_func = partial(gimo_collate_fn, dataset=test_dataset, num_sample_points=config.sample_points)
    eval_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No shuffling for evaluation
        num_workers=args.num_workers,
        drop_last=False, # Process all samples
        collate_fn=eval_collate_func
    )
    print(f"Test DataLoader ready with {len(test_dataset)} samples.")

    # --- Evaluation Loop ---
    total_l1 = 0.0
    total_rmse_ade = 0.0 # Accumulate RMSE/ADE (L2)
    total_fde = 0.0
    num_batches = 0
    total_valid_samples = 0 # Count total valid samples across all batches
    
    # For visualization
    vis_output_dir = os.path.join(args.output_dir, "visualizations")
    if args.visualize:
        os.makedirs(vis_output_dir, exist_ok=True)

    with torch.no_grad():
        progress_bar = tqdm(eval_loader, desc="Evaluating")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # --- Prepare Batch Data ---
                full_trajectory_batch = batch['full_poses'].float().to(device)  # Use full_poses (position + orientation)
                point_cloud_batch = batch['point_cloud'].float().to(device)
                bbox_corners_batch = batch['bbox_corners'].float().to(device) # Get bbox corners
                full_attention_mask = batch['full_attention_mask'].to(device)
                # Get normalization params (assuming they are added to the batch by dataset/collate)
                # Need to handle potential variations in how normalization info is stored/batched
                normalization_params = batch.get('normalization')
                if normalization_params:
                     # If normalization params are batched (e.g., list of dicts), process per item later
                     # If they are tensors (e.g., stacked), ensure they are on device
                     if isinstance(normalization_params, dict):
                         for k in normalization_params:
                             if isinstance(normalization_params[k], torch.Tensor):
                                 normalization_params[k] = normalization_params[k].to(device)
                else:
                    # Fallback if normalization params not in batch (might happen with older data)
                    print(f"Warning: Normalization parameters not found in batch {batch_idx}. Cannot denormalize.")
                    # Create dummy params indicating no normalization happened
                    normalization_params = {'is_normalized': False}
                    
                # Get object names/IDs if needed for saving visualizations
                object_names = batch.get('object_name', [])
                segment_indices = batch.get('segment_idx', [])
                
                # Get object categories for the model
                # Get object category IDs (using mapped dense IDs, not strings)
                object_category_ids = batch.get('object_category_id', None)
                if object_category_ids is not None:
                    object_category_ids = object_category_ids.to(device)
                
                # For backward compatibility and visualization purposes, also get category strings
                object_categories = batch.get('object_category', [f"unknown" for i in range(full_trajectory_batch.shape[0])])
                # Convert categories to list of strings if they're tensors
                if isinstance(object_categories, torch.Tensor):
                    object_categories = [cat.item() if isinstance(cat.item(), str) else str(cat.item()) for cat in object_categories]

                # Prepare input trajectory for the model based on config
                if config.use_first_frame_only:
                    # Use only the first frame as input
                    input_trajectory_batch = full_trajectory_batch[:, 0:1, :]
                    bbox_corners_input_batch = bbox_corners_batch[:, 0:1, :, :]  # Also slice bbox corners
                else:
                    # Use a fixed portion of the history as input
                    fixed_history_length = int(np.floor(full_trajectory_batch.shape[1] * config.history_fraction))
                    input_trajectory_batch = full_trajectory_batch[:, :fixed_history_length, :]
                    bbox_corners_input_batch = bbox_corners_batch[:, :fixed_history_length, :, :]  # Also slice bbox corners

                # --- Model Inference ---
                predicted_full_trajectory = model(
                    input_trajectory=input_trajectory_batch,
                    point_cloud=point_cloud_batch,
                    bounding_box_corners=bbox_corners_input_batch,
                    object_category_ids=object_category_ids
                )
                total_loss, loss_dict = model.compute_loss(predicted_full_trajectory, batch)

                # --- Process Each Sample in Batch ---
                batch_l1 = 0.0
                batch_rmse_ade = 0.0
                batch_fde = 0.0
                valid_samples_in_batch = 0

                for i in range(full_trajectory_batch.shape[0]):
                    # Extract full poses (positions + orientations)
                    gt_full_poses = full_trajectory_batch[i]
                    pred_full_poses = predicted_full_trajectory[i]
                    mask_full = full_attention_mask[i]
                    
                    # Extract position component (first 3 dimensions) for metrics
                    position_dim = 3  # First 3 dimensions are positions
                    gt_full = gt_full_poses[:, :position_dim]
                    pred_full = pred_full_poses[:, :position_dim]

                    # Extract normalization params for this specific sample
                    sample_norm_params = {}
                    if isinstance(normalization_params, dict) and isinstance(normalization_params.get('is_normalized'), torch.Tensor):
                        # Params were stacked in the batch
                        sample_norm_params = {k: v[i] for k, v in normalization_params.items()}
                    elif isinstance(normalization_params, dict): # Single dict shared across batch
                        sample_norm_params = normalization_params
                    elif isinstance(normalization_params, list): # List of dicts
                         if i < len(normalization_params):
                             sample_norm_params = normalization_params[i]
                         else: # Fallback if list length mismatch
                             sample_norm_params = {'is_normalized': False}
                    else: # Fallback
                        sample_norm_params = {'is_normalized': False}

                    # Ensure 'is_normalized' exists and handle tensor case
                    is_normalized = sample_norm_params.get('is_normalized', False)
                    if isinstance(is_normalized, torch.Tensor):
                        is_normalized = is_normalized.item()
                    sample_norm_params['is_normalized'] = is_normalized # Store boolean value

                    # Dynamic split based on actual length
                    actual_length = torch.sum(mask_full).int().item()
                    if actual_length < 2: continue # Skip trajectories that are too short

                    # --- Determine history length based on config ---
                    if config.use_first_frame_only:
                        dynamic_history_length = 1 if actual_length >= 1 else 0 # Past is only the first frame
                    else:
                        dynamic_history_length = int(np.floor(actual_length * config.history_fraction))
                        dynamic_history_length = max(1, min(dynamic_history_length, actual_length - 1)) # Ensure at least 1 future point
                    # ---------------------------------------------

                    # Split GT and Prediction (using the determined history length)
                    gt_hist = gt_full[:dynamic_history_length]
                    gt_future = gt_full[dynamic_history_length:actual_length]
                    pred_hist = pred_full[:dynamic_history_length]
                    pred_future = pred_full[dynamic_history_length:actual_length] # Slice prediction same as GT
                    
                    # Also extract orientation data for visualization
                    gt_hist_ori = gt_full_poses[:dynamic_history_length, position_dim:]
                    gt_future_ori = gt_full_poses[dynamic_history_length:actual_length, position_dim:]
                    pred_hist_ori = pred_full_poses[:dynamic_history_length, position_dim:]
                    pred_future_ori = pred_full_poses[dynamic_history_length:actual_length, position_dim:]
                    
                    # Get Masks
                    hist_mask = mask_full[:dynamic_history_length]
                    future_mask = mask_full[dynamic_history_length:actual_length]

                    if future_mask.sum() == 0: continue # Skip if no valid future points

                    # Denormalize if necessary
                    if is_normalized:
                        # Check if params are tensors before denormalizing
                        if not all(isinstance(sample_norm_params.get(k), torch.Tensor) for k in ['scene_min', 'scene_max', 'scene_scale']):
                             print(f"Warning: Skipping denormalization for sample {i} in batch {batch_idx} due to missing/invalid normalization tensors.")
                             continue # Skip this sample if params are invalid

                        gt_future_denorm = denormalize_trajectory(gt_future, sample_norm_params)
                        pred_future_denorm = denormalize_trajectory(pred_future, sample_norm_params)
                        
                        # Also denormalize history for visualization and potential first-frame metrics
                        gt_hist_denorm = denormalize_trajectory(gt_hist, sample_norm_params)
                        pred_hist_denorm = denormalize_trajectory(pred_hist, sample_norm_params)
                        
                        # For orientations, no normalization was applied, just copy the values
                        gt_hist_ori_denorm = gt_hist_ori
                        gt_future_ori_denorm = gt_future_ori
                        pred_hist_ori_denorm = pred_hist_ori
                        pred_future_ori_denorm = pred_future_ori
                    else:
                        gt_future_denorm = gt_future
                        pred_future_denorm = pred_future
                        gt_hist_denorm = gt_hist
                        pred_hist_denorm = pred_hist
                        
                        # For orientations, just copy the values
                        gt_hist_ori_denorm = gt_hist_ori
                        gt_future_ori_denorm = gt_future_ori
                        pred_hist_ori_denorm = pred_hist_ori
                        pred_future_ori_denorm = pred_future_ori

                    # Compute Metrics for this sample
                    l1_mean, rmse_ade, fde = compute_metrics_for_sample(pred_future_denorm, gt_future_denorm, future_mask)
                    
                    # For first_frame_only mode, also check reconstruction of the first frame
                    first_frame_rec_error = 0.0
                    if config.use_first_frame_only and gt_hist.shape[0] > 0 and hist_mask.sum() > 0:
                        # Calculate first frame reconstruction error
                        # This aligns with our model update for reconstruction loss in first_frame_only mode
                        first_frame_gt = gt_hist_denorm[0]
                        first_frame_pred = pred_hist_denorm[0]
                        first_frame_diff = torch.abs(first_frame_gt - first_frame_pred).mean().item()
                        first_frame_rec_error = first_frame_diff
                        # Optionally log first frame error
                        if batch_idx == 0 and i == 0:  # Just log for the first sample
                            print(f"First frame reconstruction error: {first_frame_rec_error:.4f}")

                    batch_l1 += l1_mean.item()
                    batch_rmse_ade += rmse_ade.item()
                    batch_fde += fde.item()
                    valid_samples_in_batch += 1

                    # --- Visualization ---
                    if args.visualize:
                         try:
                             obj_name = object_names[i] if i < len(object_names) else f"obj_{batch_idx}_{i}"
                             seg_idx = segment_indices[i].item() if i < len(segment_indices) and segment_indices[i].item() != -1 else None

                             if seg_idx is not None:
                                 filename_base = f"{obj_name}_seg{seg_idx}"
                                 vis_title_base = f"{obj_name} (Seg: {seg_idx})"
                             else:
                                 filename_base = f"{obj_name}"
                                 vis_title_base = f"{obj_name}"
                             
                             # Standard prediction vs ground truth visualization
                             pred_vs_gt_path = os.path.join(vis_output_dir, f"{filename_base}_prediction_vs_gt.png")
                              
                             visualize_prediction(
                                 past_positions=gt_hist_denorm.cpu(),
                                 future_positions_gt=gt_future_denorm.cpu(),
                                 future_positions_pred=pred_future_denorm.cpu(),
                                 past_mask=hist_mask.cpu(),
                                 future_mask_gt=future_mask.cpu(), # Use the future mask
                                 title=f"Pred vs GT - {vis_title_base} (Eval)",
                                 save_path=pred_vs_gt_path,
                                 segment_idx=seg_idx,
                                 show_orientation=args.show_ori_arrows,
                                 past_orientations=gt_hist_ori_denorm.cpu(),
                                 future_orientations_gt=gt_future_ori_denorm.cpu(),
                                 future_orientations_pred=pred_future_ori_denorm.cpu()
                             )
                             
                             # Add full trajectory visualization with bounding boxes if requested
                             if args.visualize_bbox:
                                 # Extract full GT positions and mask for visualization
                                 gt_full_denorm = torch.cat([gt_hist_denorm, gt_future_denorm], dim=0)[:actual_length]
                                 full_mask = torch.cat([hist_mask, future_mask], dim=0)[:actual_length]
                                 
                                 # Create path for full trajectory visualization
                                 full_traj_path = os.path.join(vis_output_dir, f"{filename_base}_full_trajectory_with_bbox.png")
                                 
                                 # Get the sample's bounding box corners
                                 sample_bbox_corners = bbox_corners_batch[i, :actual_length].cpu()
                                 
                                 # Denormalize bounding box corners if the data is normalized
                                 if sample_norm_params.get('is_normalized', False):
                                     # Reshape bbox corners for denormalization [T, 8, 3] -> [T*8, 3]
                                     bbox_shape = sample_bbox_corners.shape
                                     sample_bbox_corners_flat = sample_bbox_corners.reshape(-1, 3)
                                     
                                     # Denormalize
                                     sample_bbox_corners_flat_denorm = denormalize_trajectory(
                                         sample_bbox_corners_flat, 
                                         sample_norm_params
                                     )
                                     
                                     # Reshape back to original shape
                                     sample_bbox_corners = sample_bbox_corners_flat_denorm.reshape(bbox_shape)
                                 
                                 # Visualize the full trajectory with bbox
                                 visualize_full_trajectory(
                                     positions=gt_full_denorm,
                                     attention_mask=full_mask,
                                     point_cloud=point_cloud_batch[i].cpu(),  # Use point cloud for this sample
                                     bbox_corners_sequence=sample_bbox_corners,  # Add bounding box corners
                                     title=f"Full Trajectory - {vis_title_base} (Eval)",
                                     save_path=full_traj_path,
                                     segment_idx=seg_idx
                                 )
                             
                         except Exception as vis_e:
                              print(f"Warning: Error during visualization for sample {i} in batch {batch_idx}: {vis_e}")


                # --- Aggregate Batch Metrics ---
                if valid_samples_in_batch > 0:
                    # Accumulate sums weighted by valid samples in batch
                    total_l1 += batch_l1 # batch_l1 is already sum of means for the batch
                    total_rmse_ade += batch_rmse_ade # Accumulate sum of sample RMSEs
                    total_fde += batch_fde # batch_fde is already sum of means for the batch
                    total_valid_samples += valid_samples_in_batch
                    num_batches += 1
                    progress_bar.set_postfix({
                         'Batch L1': f"{(batch_l1 / valid_samples_in_batch):.4f}",
                         'Batch RMSE': f"{(batch_rmse_ade / valid_samples_in_batch):.4f}",
                         'Batch FDE': f"{(batch_fde / valid_samples_in_batch):.4f}"
                    })

            except Exception as e:
                 print(f"Error processing batch {batch_idx}: {e}")
                 import traceback
                 traceback.print_exc()
                 continue # Skip batch on error

    # --- Final Metrics ---
    if total_valid_samples > 0:
        mean_l1 = total_l1 / total_valid_samples
        mean_rmse = total_rmse_ade / total_valid_samples # This is the overall mean RMSE
        mean_fde = total_fde / total_valid_samples
        print(f"--- Evaluation Complete ---")
        print(f" Model from Epoch: {best_epoch}") # Print the epoch
        print(f" Mean L1 (MAE): {mean_l1:.4f}")
        print(f" Mean RMSE: {mean_rmse:.4f}")
        print(f" Mean FDE: {mean_fde:.4f}")
        
        # Also log special metrics for first_frame_only mode
        if config.use_first_frame_only:
            print(f" Note: Model is in first_frame_only mode - metrics reflect future prediction only")
            print(f" First frame is being used for context with separate reconstruction loss")
    else:
        print("Error: No batches were successfully processed.")
        mean_l1 = float('inf')
        mean_rmse = float('inf')
        mean_fde = float('inf')

    # --- Save Results ---
    results = {
        'model_path': args.model_path,
        'best_model_epoch': best_epoch, # Add the epoch number here
        'config': vars(config), # Save config used
        'eval_args': vars(args), # Save evaluation args
        'mean_l1': mean_l1,
        'mean_rmse': mean_rmse,
        'mean_fde': mean_fde,
        'first_frame_only_mode': config.use_first_frame_only, # Flag indicating special mode
        'num_test_samples_processed': total_valid_samples, # Report processed samples
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'comment': os.path.basename(os.path.normpath(args.model_path)) # Extract comment from model_path
    }

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {results_path}")

    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    print(f"Evaluating GIMO ADT model: {args.model_path}")
    
    # Set output directory with timestamp and comment if provided
    output_dir = args.output_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Extract comment from model path's parent directory
    model_dir = os.path.dirname(os.path.normpath(args.model_path))
    comment = os.path.basename(model_dir)
    # Create timestamped output dir
    output_dir = f"{output_dir}/{timestamp}_{comment}"
    args.output_dir = output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and config
    try:
        model, config, best_epoch = load_model_and_config(args.model_path, device)
        
        # Override config with command line arguments if specified
        if args.use_first_frame_only:
            print(f"Overriding config: use_first_frame_only set to True (from command line)")
            config.use_first_frame_only = True
            
        # Note: show_ori_arrows will be used directly from args in the evaluate function
            
        # --- Print Model Architecture Details ---
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n=== MODEL ARCHITECTURE ===")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Text Embedding Enabled: {not getattr(config, 'no_text_embedding', False)}")
        print(f"Use First Frame Only: {config.use_first_frame_only}")
        
        # Log model components
        components = {
            'Motion Linear': (model.motion_linear, []), 
            'Scene Encoder': (model.scene_encoder, ['hparams']), 
            'FP Layer': (model.fp_layer, []),
            'BBox PointNet': (model.bbox_pointnet, ['conv1']), 
            'Motion BBox Encoder': (model.motion_bbox_encoder, ['n_input_channels', 'n_latent_channels', 'n_self_att_heads', 'n_self_att_layers']),
            'Embedding Layer (Fusion 2)': (model.embedding_layer, ['in_features', 'out_features']),
            'Output Encoder': (model.output_encoder, ['n_input_channels', 'n_latent_channels', 'n_self_att_heads', 'n_self_att_layers']),
            'Output Layer': (model.outputlayer, ['in_features', 'out_features'])
        }
        
        # Add text embedding components if enabled
        if not getattr(config, 'no_text_embedding', False): # Check if text embedding is enabled
            if hasattr(model, 'category_embedding'): # Ensure the attribute exists
                components.update({
                    'Category Embedding': (model.category_embedding, ['num_embeddings', 'embedding_dim'])
                })
        
        # Log each component's structure and parameters
        for name, (component, attrs) in components.items():
            params = sum(p.numel() for p in component.parameters())
            print(f"\n{name}:")
            print(f"  Parameters: {params:,}")
            try:
                # Try to log attributes if available
                for attr in attrs:
                    if hasattr(component, attr):
                        print(f"  {attr}: {getattr(component, attr)}")
            except:
                pass
        
        print("=== END MODEL ARCHITECTURE ===\n")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run evaluation
    evaluate(model, config, args, best_epoch)

if __name__ == "__main__":
    main()

# Example usage:
# 1. Basic evaluation with visualization
# python evaluate_gimo_adt.py --model_path ./checkpoints/best_model.pth --visualize --num_vis_samples 5
#
# 2. Evaluating a model trained with use_first_frame_only enabled
# python evaluate_gimo_adt.py --model_path ./checkpoints/first_frame_only_model.pth --visualize --num_vis_samples 10 --output_dir eval_first_frame_only
#
# 3. Evaluating on a specific test set (overriding the one in the checkpoint's config)
# python evaluate_gimo_adt.py --model_path ./checkpoints/gimo_model.pth --adt_dataroot /path/to/test_data --batch_size 8 
#
# 4. Example for latest GIMO model with text categories
# python evaluate_gimo_adt.py --model_path ./checkpoints/gimo_adt_with_categories.pth --visualize --num_vis_samples 20 --output_dir eval_with_categories
#
# 5. Evaluating 6D pose model with orientation visualization enabled
# python evaluate_gimo_adt.py --model_path ./checkpoints/full_6d_model.pth --visualize --num_vis_samples 30 --output_dir eval_with_orientation 
#
# 6. Using the new command line arguments to override config
# python evaluate_gimo_adt.py --model_path ./checkpoints/best_model.pth --visualize --show_ori_arrows --use_first_frame_only 