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
import logging
from functools import partial # Import partial for binding args to collate_fn
from torch.utils.data.dataloader import default_collate # Import default_collate
import shutil # Import shutil

# --- Imports from our project ---
from model.gimo_adt_model import GIMO_ADT_Model
from dataset.gimo_multi_sequence_dataset import GIMOMultiSequenceDataset
from dataset.gimo_adt_trajectory_dataset import GimoAriaDigitalTwinTrajectoryDataset # Needed for denormalization access potentially
from config.adt_config import ADTObjectMotionConfig
from utils.visualization import visualize_trajectory, visualize_prediction, visualize_full_trajectory # Import visualization utils
from utils.rerun_visualization import (
    initialize_rerun,
    downsample_point_cloud,
    extract_trajectory_specific_point_cloud,
    visualize_trajectory_rerun,
    save_rerun_recording,
    HAS_RERUN
)
import rerun as rr

# Import metrics utilities
from utils.metrics_utils import (
    transform_coords_for_visualization,
    compute_metrics_for_sample,
    gimo_collate_fn
)

def setup_logger(output_dir):
    """Set up logger to both write to console and to file."""
    # Create a logger
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    log_file = os.path.join(output_dir, 'evaluation.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log start information
    logger.info("=" * 50)
    logger.info(f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    return logger

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
        default=1, # Can override config batch size for evaluation
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
        default=500, # Number of trajectory samples to visualize
        help="Number of trajectory samples to visualize"
    )
    parser.add_argument(
        "--visualize_bbox",
        action="store_true",
        help="Enable visualization of bounding boxes"
    )
    
    # --- Rerun Visualization Options ---
    parser.add_argument(
        "--use_rerun",
        action="store_true",
        help="Enable Rerun visualization"
    )
    parser.add_argument(
        "--pointcloud_downsample_factor",
        type=int,
        default=1,
        help="Factor for downsampling point clouds in Rerun visualization"
    )
    parser.add_argument(
        "--rerun_line_width",
        type=float,
        default=0.02,
        help="Width of lines in Rerun visualization"
    )
    parser.add_argument(
        "--rerun_point_size",
        type=float,
        default=0.03,
        help="Size of points in Rerun visualization"
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
    parser.add_argument(
        "--num_top_worst_to_save",
        type=int,
        default=10,
        help="Number of best and worst Rerun recordings and visualizations to save."
    )

    return parser.parse_args()

def load_model_and_config(checkpoint_path, device, logger):
    """
    Load GIMO_ADT_Model and its config from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pth)
        device: Device to load the model on
        logger: Logger instance for logging information
        
    Returns:
        model: Loaded GIMO_ADT_Model
        config: Configuration object from the checkpoint
        best_epoch (int): The epoch number at which this checkpoint was saved (usually the best validation epoch). Returns 0 if not found.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        error_msg = f"Checkpoint file not found: {checkpoint_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' not in checkpoint:
        error_msg = "Checkpoint does not contain 'config'. Cannot instantiate model."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if 'model_state_dict' not in checkpoint:
        error_msg = "Checkpoint does not contain 'model_state_dict'."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # Load config (convert dict back to Namespace or use directly if already object)
    config_dict = checkpoint['config']
    config = argparse.Namespace(**config_dict) # Convert dict back to Namespace
    
    logger.info("Configuration loaded from checkpoint:")
    logger.info(str(config))

    # Create model instance using loaded config
    model = GIMO_ADT_Model(config).to(device)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode
    
    param_count = sum(p.numel() for p in model.parameters())
    param_count_millions = param_count / 1_000_000
    logger.info(f"Loaded GIMO_ADT_Model with {param_count_millions:.2f}M parameters.")
    
    # Get the epoch number
    best_epoch = checkpoint.get('epoch', 0) # Default to 0 if epoch key doesn't exist
    if best_epoch > 0:
        logger.info(f"Checkpoint was saved at epoch: {best_epoch}")
    else:
        logger.warning("Warning: Epoch number not found in checkpoint.")
        
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

def evaluate(model, config, args, best_epoch, logger, 
             best_rerun_dir, best_vis_dir, worst_rerun_dir, worst_vis_dir):
    """
    Run the evaluation loop.
    
    Args:
        model: The loaded GIMO_ADT_Model.
        config: The configuration object (usually loaded from checkpoint).
        args: Command line arguments for evaluation.
        best_epoch (int): The epoch number the model checkpoint was saved at.
        logger: Logger instance for logging information.
        best_rerun_dir: Directory to save best Rerun recordings
        best_vis_dir: Directory to save best visualizations
        worst_rerun_dir: Directory to save worst Rerun recordings
        worst_vis_dir: Directory to save worst visualizations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # --- Setup Dataset and DataLoader ---
    logger.info("Setting up test dataset...")
    # Use config values from checkpoint, potentially overridden by args
    test_sequences = None
    checkpoint_dir = os.path.dirname(args.model_path)
    val_split_path = os.path.join(checkpoint_dir, 'val_sequences.txt')
    dataroot_override = args.adt_dataroot # Store override if provided
    
    if os.path.exists(val_split_path):
        try:
            logger.info(f"Loading test sequences from saved validation split: {val_split_path}")
            with open(val_split_path, 'r') as f:
                test_sequences = [line.strip() for line in f if line.strip()]
            if not test_sequences:
                 logger.warning(f"Warning: {val_split_path} is empty.")
                 test_sequences = None # Force fallback
            else:
                 logger.info(f"Loaded {len(test_sequences)} test sequences.")
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
                                 logger.warning(f"Warning: Could not adjust path for {seq} using override {dataroot_override}")
                                 adjusted_sequences.append(seq) # Keep original if adjustment fails
                          else:
                              adjusted_sequences.append(seq)
                      test_sequences = adjusted_sequences
                      logger.info(f"Adjusted sequence paths using dataroot override: {dataroot_override}")
                             
        except Exception as e:
            logger.warning(f"Warning: Error loading {val_split_path}: {e}. Will attempt fallback.")
            test_sequences = None
    else:
        logger.warning(f"Warning: val_sequences.txt not found in {checkpoint_dir}. Attempting fallback using config.")

    # Fallback logic if val_sequences.txt was not loaded successfully
    if test_sequences is None:
        logger.info("Attempting fallback: Using adt_dataroot from config/args...")
        dataroot = dataroot_override if dataroot_override is not None else config.adt_dataroot
        if os.path.isdir(dataroot):
             # Use all sequences in dataroot as test set (assuming no split info available)
             logger.warning(f"Warning: Using ALL sequences found in {dataroot} as test set due to missing split info.")
             # Import sequence utils dynamically if needed for fallback
             try:
                 from ariaworldgaussians.adt_sequence_utils import find_adt_sequences
                 test_sequences = find_adt_sequences(dataroot)
             except ImportError:
                 logger.warning("Warning: Cannot import find_adt_sequences. Manually scanning directory.")
                 test_sequences = [os.path.join(dataroot, item) for item in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, item))]
                 
             if not test_sequences:
                  logger.error(f"Error: No sequences found in fallback dataroot {dataroot}.")
                  return None
             logger.info(f"Found {len(test_sequences)} sequences in fallback dataroot.")
        elif os.path.exists(dataroot): # Check if dataroot itself is a single sequence path
             test_sequences = [dataroot]
             logger.info(f"Using single sequence {dataroot} as test set (fallback).")
        else:
             logger.error(f"Error: Fallback failed. Dataroot path {dataroot} not found or invalid.")
             return None

    # Use cache dir based on output dir to avoid conflicts with training cache
    # Prioritize global_cache_dir if provided
    if args.global_cache_dir:
        eval_cache_dir = args.global_cache_dir
        os.makedirs(eval_cache_dir, exist_ok=True)
        logger.info(f"Using global cache directory for evaluation: {eval_cache_dir}")
    else:
        eval_cache_dir = os.path.join(args.output_dir, 'trajectory_cache_eval')
        os.makedirs(eval_cache_dir, exist_ok=True)
        logger.info(f"Using evaluation-specific cache directory: {eval_cache_dir}")

    test_dataset = GIMOMultiSequenceDataset(
        sequence_paths=test_sequences,
        config=config, # Use loaded config
        cache_dir=eval_cache_dir, # Use separate cache for eval
        use_cache=True # Enable caching for evaluation
    )

    if len(test_dataset) == 0:
        logger.error("Error: Test dataset is empty. Check test split file and sequence paths.")
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
    logger.info(f"Test DataLoader ready with {len(test_dataset)} samples.")
    # --- Evaluation Loop ---
    total_l1 = 0.0
    total_rmse_ade = 0.0 # Accumulate RMSE/ADE (L2)
    total_fde = 0.0
    num_batches = 0
    total_valid_samples = 0 # Count total valid samples across all batches
    all_samples_metrics = [] # List to store metrics and paths for each sample
    
    # Track visualization statistics
    visualized_count = 0
    skipped_visualizations = 0
    
    # For Matplotlib visualization
    vis_output_dir = os.path.join(args.output_dir, "visualizations")
    if args.visualize:
        os.makedirs(vis_output_dir, exist_ok=True)
        logger.info(f"Matplotlib visualizations will be saved to {vis_output_dir}")

    # Base directory for per-sample Rerun .rrd files
    # This will be created if Rerun is used.
    per_sample_rrd_basedir = None
    if args.use_rerun and HAS_RERUN:
        per_sample_rrd_basedir = os.path.join(args.output_dir, "rerun_visualizations_per_sample")
        os.makedirs(per_sample_rrd_basedir, exist_ok=True)
        logger.info(f"Rerun per-sample .rrd files will be saved to: {per_sample_rrd_basedir}")

    with torch.no_grad():
        progress_bar = tqdm(eval_loader, desc="Evaluating")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                logger.info(f"Processing batch {batch_idx+1}/{len(eval_loader)}")
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
                    logger.warning(f"Warning: Normalization parameters not found in batch {batch_idx}. Cannot denormalize.")
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

                # --- Dynamically determine input history length for the batch (evaluation) ---
                actual_lengths_batch = full_attention_mask.sum(dim=1).int() # Shape [B]

                if config.use_first_frame_only:
                    hist_len_for_batch = 1 if torch.any(actual_lengths_batch > 0) else 0
                    input_trajectory_batch = full_trajectory_batch[:, :hist_len_for_batch, :]
                    if not config.no_bbox and bbox_corners_batch is not None:
                        bbox_corners_input_batch = bbox_corners_batch[:, :hist_len_for_batch, :, :]
                    else:
                        bbox_corners_input_batch = None
                else:
                    min_val_tensor = torch.ones_like(actual_lengths_batch).float()
                    individual_dynamic_hist_lengths = (actual_lengths_batch.float() * config.history_fraction).floor().long()
                    individual_dynamic_hist_lengths = torch.max(min_val_tensor.long(), individual_dynamic_hist_lengths)
                    individual_dynamic_hist_lengths = torch.min(individual_dynamic_hist_lengths, actual_lengths_batch)
                    individual_dynamic_hist_lengths[actual_lengths_batch == 0] = 0
                    
                    hist_len_for_batch = torch.max(individual_dynamic_hist_lengths).item()
                    # Ensure hist_len_for_batch does not exceed the padded sequence length
                    hist_len_for_batch = min(hist_len_for_batch, full_trajectory_batch.shape[1])

                    input_trajectory_batch = full_trajectory_batch[:, :hist_len_for_batch, :]
                    if not config.no_bbox and bbox_corners_batch is not None:
                        bbox_corners_input_batch = bbox_corners_batch[:, :hist_len_for_batch, :, :]
                    else:
                        bbox_corners_input_batch = None
                # --- End dynamic input history length determination (evaluation) ---

                logger.info(f"Input trajectory shape: {input_trajectory_batch.shape}, Full trajectory shape: {full_trajectory_batch.shape}")
                
                # --- Model Inference ---
                logger.info("Running model inference...")
                predicted_full_trajectory = model(
                    input_trajectory=input_trajectory_batch,
                    point_cloud=point_cloud_batch,
                    bounding_box_corners=bbox_corners_input_batch,
                    object_category_ids=object_category_ids
                )
                total_loss, loss_dict = model.compute_loss(predicted_full_trajectory, batch)
                logger.info(f"Loss: {total_loss:.4f}")

                # --- Process Each Sample in Batch ---
                batch_l1 = 0.0
                batch_rmse_ade = 0.0
                batch_fde = 0.0
                valid_samples_in_batch = 0

                logger.info(f"Processing {full_trajectory_batch.shape[0]} samples in batch")
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
                    if actual_length < 2:
                        logger.info(f"Sample {i} skipped: trajectory too short (length={actual_length})")
                        continue # Skip trajectories that are too short

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
                    
                    # Also extract rotation data for visualization
                    gt_hist_rot = gt_full_poses[:dynamic_history_length, position_dim:]
                    gt_future_rot = gt_full_poses[dynamic_history_length:actual_length, position_dim:]
                    pred_hist_rot = pred_full_poses[:dynamic_history_length, position_dim:]
                    pred_future_rot = pred_full_poses[dynamic_history_length:actual_length, position_dim:]
                    
                    # Get Masks
                    hist_mask = mask_full[:dynamic_history_length]
                    future_mask = mask_full[dynamic_history_length:actual_length]

                    if future_mask.sum() == 0:
                        logger.info(f"Sample {i} skipped: no valid future points")
                        continue # Skip if no valid future points

                    # Denormalize if necessary
                    if is_normalized:
                        # Check if params are tensors before denormalizing
                        if not all(isinstance(sample_norm_params.get(k), torch.Tensor) for k in ['scene_min', 'scene_max', 'scene_scale']):
                             logger.warning(f"Warning: Skipping denormalization for sample {i} in batch {batch_idx} due to missing/invalid normalization tensors.")
                             continue # Skip this sample if params are invalid

                        gt_future_denorm = denormalize_trajectory(gt_future, sample_norm_params)
                        pred_future_denorm = denormalize_trajectory(pred_future, sample_norm_params)
                        
                        # Also denormalize history for visualization and potential first-frame metrics
                        gt_hist_denorm = denormalize_trajectory(gt_hist, sample_norm_params)
                        pred_hist_denorm = denormalize_trajectory(pred_hist, sample_norm_params)
                        
                        # For rotations, no normalization was applied, just copy the values
                        gt_hist_rot_denorm = gt_hist_rot
                        gt_future_rot_denorm = gt_future_rot
                        pred_hist_rot_denorm = pred_hist_rot
                        pred_future_rot_denorm = pred_future_rot
                    else:
                        gt_future_denorm = gt_future
                        pred_future_denorm = pred_future
                        gt_hist_denorm = gt_hist
                        pred_hist_denorm = pred_hist
                        
                        # For rotations, just copy the values
                        gt_hist_rot_denorm = gt_hist_rot
                        gt_future_rot_denorm = gt_future_rot
                        pred_hist_rot_denorm = pred_hist_rot
                        pred_future_rot_denorm = pred_future_rot

                    # Compute Metrics for this sample
                    l1_mean, rmse_ade, fde = compute_metrics_for_sample(pred_future_denorm, gt_future_denorm, future_mask)
                    logger.info(f"Sample {i}: L1={l1_mean.item():.4f}, RMSE={rmse_ade.item():.4f}, FDE={fde.item():.4f}")
                    
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
                            logger.info(f"First frame reconstruction error: {first_frame_rec_error:.4f}")

                    batch_l1 += l1_mean.item()
                    batch_rmse_ade += rmse_ade.item()
                    batch_fde += fde.item()
                    valid_samples_in_batch += 1

                    # --- Store sample metrics and paths for later sorting --- 
                    # Ensure paths are captured correctly, especially for Rerun
                    current_sample_rerun_path = None
                    if args.use_rerun and HAS_RERUN and per_sample_rrd_basedir: # Check if Rerun was used for this sample
                        # Construct the expected path if Rerun was initialized and saved
                        # filename_base was defined earlier in the visualization section
                        if 'filename_base' in locals():
                            current_sample_rerun_path = os.path.join(per_sample_rrd_basedir, f"{filename_base}.rrd")
                            if not os.path.exists(current_sample_rerun_path):
                                # This check is crucial. If the file wasn't saved as expected, log it.
                                logger.warning(f"Rerun file {current_sample_rerun_path} not found for sample {filename_base}. It will not be copied.")
                                current_sample_rerun_path = None # Reset if not found
                        else:
                            logger.warning("filename_base not defined when trying to capture Rerun path.")
                    
                    current_sample_vis_path = None
                    if args.visualize and 'pred_vs_gt_path' in locals() and os.path.exists(pred_vs_gt_path):
                        current_sample_vis_path = pred_vs_gt_path
                    elif args.visualize:
                        logger.warning(f"Visualization path {pred_vs_gt_path if 'pred_vs_gt_path' in locals() else 'unknown'} not found or not generated.")

                    all_samples_metrics.append({
                        'l1': l1_mean.item(),
                        'rmse': rmse_ade.item(),
                        'fde': fde.item(),
                        'vis_path': current_sample_vis_path, # Path to matplotlib vis
                        'rerun_path': current_sample_rerun_path,  # Path to Rerun .rrd file
                        'sample_name': filename_base if 'filename_base' in locals() else f"batch{batch_idx}_sample{i}"
                    })
                    # ----------------------------------------------------------

                    # --- Visualization ---
                    if args.visualize:
                         try:
                             # Check if we've reached visualization limit
                             if visualized_count >= args.num_vis_samples:
                                 skipped_visualizations += 1
                                 continue
                                 
                             obj_name = object_names[i] if i < len(object_names) else f"obj_{batch_idx}_{i}"
                             seg_idx = segment_indices[i].item() if i < len(segment_indices) and segment_indices[i].item() != -1 else None

                             # --- MODIFICATION START for filename and title ---
                             current_sequence_path = batch['sequence_path'][i] # Get sequence path for the current sample
                             sequence_name_extracted = os.path.splitext(os.path.basename(current_sequence_path))[0]

                             fn_obj_name = obj_name
                             fn_seq_name = f"seq_{sequence_name_extracted}"
                             fn_seg_id = f"seg{seg_idx}" if seg_idx is not None else "segNA"
                             fn_batch_id = f"batch{batch_idx}" # batch_idx is from the outer loop over dataloader

                             filename_base = f"{fn_obj_name}_{fn_seq_name}_{fn_seg_id}_{fn_batch_id}"
                             
                             title_obj_name = obj_name
                             title_seq_name = f"Seq: {sequence_name_extracted}"
                             title_seg_id = f"Seg: {seg_idx if seg_idx is not None else 'NA'}"
                             title_batch_id = f"Batch: {batch_idx}"

                             vis_title_base = f"{title_obj_name} ({title_seq_name}, {title_seg_id}, {title_batch_id})"
                             # --- MODIFICATION END ---
                             
                             logger.info(f"Visualizing sample {i}: {vis_title_base}")
                             
                             # --- Apply coordinate transformation for visualization ---
                             gt_hist_denorm_vis = transform_coords_for_visualization(gt_hist_denorm.cpu())
                             gt_future_denorm_vis = transform_coords_for_visualization(gt_future_denorm.cpu())
                             pred_future_denorm_vis = transform_coords_for_visualization(pred_future_denorm.cpu())
                             
                             # Standard prediction vs ground truth visualization
                             pred_vs_gt_path = os.path.join(vis_output_dir, f"{filename_base}_prediction_vs_gt.png")
                              
                             visualize_prediction(
                                 past_positions=gt_hist_denorm_vis,
                                 future_positions_gt=gt_future_denorm_vis,
                                 future_positions_pred=pred_future_denorm_vis,
                                 past_mask=hist_mask.cpu(),
                                 future_mask_gt=future_mask.cpu(), # Use the future mask
                                 title=f"Pred vs GT - {vis_title_base} (Eval)",
                                 save_path=pred_vs_gt_path,
                                 segment_idx=seg_idx,
                                 show_orientation=args.show_ori_arrows,
                                 past_orientations=gt_hist_rot_denorm.cpu(), # Rotations not transformed for now
                                 future_orientations_gt=gt_future_rot_denorm.cpu(), # Rotations not transformed for now
                                 future_orientations_pred=pred_future_rot_denorm.cpu() # Rotations not transformed for now
                             )
                             
                             # Add full trajectory visualization with bounding boxes if requested
                             if args.visualize_bbox:
                                 # Extract full GT positions and mask for visualization
                                 gt_full_denorm = torch.cat([gt_hist_denorm, gt_future_denorm], dim=0)[:actual_length]
                                 full_mask = torch.cat([hist_mask, future_mask], dim=0)[:actual_length]
                                 
                                 # Create path for full trajectory visualization
                                 full_traj_path = os.path.join(vis_output_dir, f"{filename_base}_full_trajectory_with_bbox.png")
                                 
                                 # Get the sample's bounding box corners
                                 sample_bbox_corners_cpu = bbox_corners_batch[i, :actual_length].cpu()
                                 
                                 # Denormalize bounding box corners if the data is normalized
                                 if sample_norm_params.get('is_normalized', False):
                                     # Reshape bbox corners for denormalization [T, 8, 3] -> [T*8, 3]
                                     bbox_shape = sample_bbox_corners_cpu.shape
                                     sample_bbox_corners_flat = sample_bbox_corners_cpu.reshape(-1, 3)
                                     
                                     # Denormalize
                                     sample_bbox_corners_flat_denorm = denormalize_trajectory(
                                         sample_bbox_corners_flat, 
                                         sample_norm_params
                                     )
                                     
                                     # Reshape back to original shape
                                     sample_bbox_corners_cpu = sample_bbox_corners_flat_denorm.reshape(bbox_shape)
                                 
                                 # Apply transformation for visualization
                                 gt_full_denorm_vis = transform_coords_for_visualization(gt_full_denorm.cpu())
                                 sample_point_cloud_vis = transform_coords_for_visualization(point_cloud_batch[i].cpu())
                                 sample_bbox_corners_vis = transform_coords_for_visualization(sample_bbox_corners_cpu)

                                 # Visualize the full trajectory with bbox
                                 visualize_full_trajectory(
                                     positions=gt_full_denorm_vis,
                                     attention_mask=full_mask,
                                     point_cloud=sample_point_cloud_vis,  # Use point cloud for this sample
                                     bbox_corners_sequence=sample_bbox_corners_vis,  # Add bounding box corners
                                     title=f"Full Trajectory - {vis_title_base} (Eval)",
                                     save_path=full_traj_path,
                                     segment_idx=seg_idx
                                 )
                             
                             # --- Rerun Visualization (per sample) ---
                             if args.use_rerun and HAS_RERUN and per_sample_rrd_basedir:
                                 try:
                                     # --- Construct unique names and paths for this sample's Rerun recording ---
                                     sample_recording_name = filename_base # e.g., "obj_seq_seg_batch"
                                     logger.info(f"Attempting Rerun visualization for sample: {sample_recording_name}")

                                     # Initialize Rerun for THIS SPECIFIC SAMPLE.
                                     sample_rerun_initialized, returned_sample_rrd_path = initialize_rerun(
                                         recording_name=sample_recording_name, 
                                         spawn=False, # No viewer per sample
                                         output_dir=per_sample_rrd_basedir 
                                     )

                                     if sample_rerun_initialized:
                                         # --- Prepare point cloud for Rerun (moved here, was missing) ---
                                         traj_point_cloud = None # Initialize to None
                                         if point_cloud_batch is not None and point_cloud_batch.shape[0] > i:
                                             sample_pc_for_rerun = point_cloud_batch[i].cpu()
                                             if args.pointcloud_downsample_factor > 1:
                                                 sample_pc_for_rerun = downsample_point_cloud(
                                                     sample_pc_for_rerun, 
                                                     args.pointcloud_downsample_factor
                                                 )
                                             traj_point_cloud = sample_pc_for_rerun
                                         
                                         # Check for trajectory-specific point cloud from the batch (if applicable)
                                         if 'trajectory_specific_pointcloud' in batch and i < len(batch['trajectory_specific_pointcloud']):
                                             traj_specific_pc_data = batch['trajectory_specific_pointcloud'][i]
                                             if traj_specific_pc_data is not None and isinstance(traj_specific_pc_data, torch.Tensor) and traj_specific_pc_data.numel() > 0:
                                                 processed_traj_specific_pc = traj_specific_pc_data.cpu()
                                                 if args.pointcloud_downsample_factor > 1:
                                                     processed_traj_specific_pc = downsample_point_cloud(
                                                         processed_traj_specific_pc, 
                                                         args.pointcloud_downsample_factor
                                                     )
                                                 traj_point_cloud = processed_traj_specific_pc # Override with more specific PC if available
                                         # --- End of point cloud preparation ---

                                         # Apply transformation for Rerun visualization
                                         gt_hist_denorm_rerun_vis = transform_coords_for_visualization(gt_hist_denorm.cpu())
                                         gt_future_denorm_rerun_vis = transform_coords_for_visualization(gt_future_denorm.cpu())
                                         pred_future_denorm_rerun_vis = transform_coords_for_visualization(pred_future_denorm.cpu())
                                         traj_point_cloud_rerun_vis = transform_coords_for_visualization(traj_point_cloud.cpu() if traj_point_cloud is not None else None)
                                         
                                         # Log data for this sample. 
                                         visualize_trajectory_rerun(
                                             past_positions=gt_hist_denorm_rerun_vis,
                                             future_positions_gt=gt_future_denorm_rerun_vis,
                                             future_positions_pred=pred_future_denorm_rerun_vis,
                                             past_mask=hist_mask,
                                             future_mask_gt=future_mask,
                                             point_cloud=traj_point_cloud_rerun_vis, # Now defined and transformed
                                             past_orientations=gt_hist_rot_denorm.cpu(), # Orientations not transformed
                                             future_orientations_gt=gt_future_rot_denorm.cpu(), # Orientations not transformed
                                             future_orientations_pred=pred_future_rot_denorm.cpu(), # Orientations not transformed
                                             object_name=obj_name, 
                                             sequence_name=sequence_name_extracted, 
                                             segment_idx=seg_idx, 
                                             line_width=args.rerun_line_width,
                                             point_size=args.rerun_point_size
                                         )

                                         # Explicitly save if init fell back
                                         if returned_sample_rrd_path: # This path is where it *should* be saved or was intended.
                                             logger.info(f"Attempting to explicitly save/finalize Rerun recording for sample {sample_recording_name} to: {returned_sample_rrd_path}")
                                             save_rerun_recording(output_path=returned_sample_rrd_path)
                                         
                                         # For now, we rely on rr.init() behavior. If data leaks between samples, consider rr.disconnect().
                                         if hasattr(rr, 'disconnect') and callable(rr.disconnect):
                                             try:
                                                 rr.disconnect()
                                                 logger.info(f"Disconnected Rerun session for sample {sample_recording_name}")
                                             except Exception as disconnect_e:
                                                 logger.warning(f"Error during Rerun disconnect for sample {sample_recording_name}: {disconnect_e}")

                                     else:
                                         logger.warning(f"Failed to initialize Rerun for sample: {sample_recording_name}. Skipping Rerun vis for this sample.")
                                 except Exception as rerun_sample_e:
                                     logger.warning(f"Warning: Error during Rerun visualization for sample {i} in batch {batch_idx} ({filename_base}): {rerun_sample_e}")
                             
                             visualized_count += 1
                             
                         except Exception as vis_e:
                              logger.warning(f"Warning: Error during visualization for sample {i} in batch {batch_idx}: {vis_e}")


                # --- Aggregate Batch Metrics ---
                if valid_samples_in_batch > 0:
                    # Accumulate sums weighted by valid samples in batch
                    total_l1 += batch_l1 # batch_l1 is already sum of means for the batch
                    total_rmse_ade += batch_rmse_ade # Accumulate sum of sample RMSEs
                    total_fde += batch_fde # batch_fde is already sum of means for the batch
                    total_valid_samples += valid_samples_in_batch
                    num_batches += 1
                    batch_avg_l1 = batch_l1 / valid_samples_in_batch
                    batch_avg_rmse = batch_rmse_ade / valid_samples_in_batch
                    batch_avg_fde = batch_fde / valid_samples_in_batch
                    
                    logger.info(f"Batch {batch_idx} metrics: L1={batch_avg_l1:.4f}, RMSE={batch_avg_rmse:.4f}, FDE={batch_avg_fde:.4f}")
                    progress_bar.set_postfix({
                         'Batch L1': f"{batch_avg_l1:.4f}",
                         'Batch RMSE': f"{batch_avg_rmse:.4f}",
                         'Batch FDE': f"{batch_avg_fde:.4f}"
                    })

            except Exception as e:
                 logger.error(f"Error processing batch {batch_idx}: {e}")
                 import traceback
                 logger.error(traceback.format_exc())
                 continue # Skip batch on error

    # --- Final Metrics ---
    if total_valid_samples > 0:
        mean_l1 = total_l1 / total_valid_samples
        mean_rmse = total_rmse_ade / total_valid_samples # This is the overall mean RMSE
        mean_fde = total_fde / total_valid_samples

        logger.info(f"--- Evaluation Complete ---")
        logger.info(f" Model from Epoch: {best_epoch}") # Print the epoch
        logger.info(f" Mean L1 (MAE): {mean_l1:.4f}")
        logger.info(f" Mean RMSE: {mean_rmse:.4f}")
        logger.info(f" Mean FDE: {mean_fde:.4f}")
        
        # Log visualization statistics
        logger.info(f" Visualized {visualized_count} samples")
        if skipped_visualizations > 0:
            logger.info(f" Skipped {skipped_visualizations} visualizations (limit of {args.num_vis_samples} reached)")
        
        # Also log special metrics for first_frame_only mode
        if config.use_first_frame_only:
            logger.info(f" Note: Model is in first_frame_only mode - metrics reflect future prediction only")
            logger.info(f" First frame is being used for context with separate reconstruction loss")
    else:
        logger.error("Error: No batches were successfully processed.")
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
        'comment': os.path.basename(os.path.normpath(args.model_path)), # Extract comment from model_path
        'num_visualized_samples': visualized_count
    }

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Evaluation results saved to {results_path}")

    # --- Sort and Save Best/Worst Reruns and Visualizations ---
    if all_samples_metrics:
        # Sort samples by L1 loss (ascending for best, descending for worst)
        all_samples_metrics.sort(key=lambda x: x['l1'])
        
        num_to_save = args.num_top_worst_to_save
        logger.info(f"Saving top/worst {num_to_save} Rerun recordings and visualizations...")

        # Save best samples
        logger.info(f"--- Saving {min(num_to_save, len(all_samples_metrics))} best samples ---")
        for i in range(min(num_to_save, len(all_samples_metrics))):
            sample_data = all_samples_metrics[i]
            logger.info(f"Best sample {i+1}/{num_to_save} (L1: {sample_data['l1']:.4f}): {sample_data['sample_name']}")
            if sample_data['vis_path'] and os.path.exists(sample_data['vis_path']):
                try:
                    shutil.copy(sample_data['vis_path'], best_vis_dir)
                    logger.info(f"  Copied visualization to {best_vis_dir}")
                except Exception as e:
                    logger.error(f"  Error copying visualization {sample_data['vis_path']}: {e}")
            if sample_data['rerun_path'] and os.path.exists(sample_data['rerun_path']):
                try:
                    shutil.copy(sample_data['rerun_path'], best_rerun_dir)
                    logger.info(f"  Copied Rerun recording to {best_rerun_dir}")
                except Exception as e:
                    logger.error(f"  Error copying Rerun recording {sample_data['rerun_path']}: {e}")

        # Save worst samples (from the end of the sorted list)
        logger.info(f"--- Saving {min(num_to_save, len(all_samples_metrics))} worst samples ---")
        for i in range(min(num_to_save, len(all_samples_metrics))):
            sample_data = all_samples_metrics[-(i+1)] # Get from the end
            logger.info(f"Worst sample {i+1}/{num_to_save} (L1: {sample_data['l1']:.4f}): {sample_data['sample_name']}")
            if sample_data['vis_path'] and os.path.exists(sample_data['vis_path']):
                try:
                    shutil.copy(sample_data['vis_path'], worst_vis_dir)
                    logger.info(f"  Copied visualization to {worst_vis_dir}")
                except Exception as e:
                    logger.error(f"  Error copying visualization {sample_data['vis_path']}: {e}")
            if sample_data['rerun_path'] and os.path.exists(sample_data['rerun_path']):
                try:
                    shutil.copy(sample_data['rerun_path'], worst_rerun_dir)
                    logger.info(f"  Copied Rerun recording to {worst_rerun_dir}")
                except Exception as e:
                    logger.error(f"  Error copying Rerun recording {sample_data['rerun_path']}: {e}")
    # ------------------------------------------------------------

    return results

def main():
    """Main evaluation function."""
    args = parse_args()
    
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
    
    # --- Create subdirectories for best/worst visualizations and Reruns ---
    best_rerun_dir = os.path.join(output_dir, "best_rerun")
    best_vis_dir = os.path.join(output_dir, "best_visualization")
    worst_rerun_dir = os.path.join(output_dir, "worst_rerun")
    worst_vis_dir = os.path.join(output_dir, "worst_visualization")
    
    os.makedirs(best_rerun_dir, exist_ok=True)
    os.makedirs(best_vis_dir, exist_ok=True)
    os.makedirs(worst_rerun_dir, exist_ok=True)
    os.makedirs(worst_vis_dir, exist_ok=True)
    # -----------------------------------------------------------------------
    
    # Setup logger
    logger = setup_logger(output_dir)
    logger.info(f"Evaluating GIMO ADT model: {args.model_path}")
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Log visualization settings
    logger.info("\n=== VISUALIZATION SETTINGS ===")
    logger.info(f"Matplotlib visualization: {'Enabled' if args.visualize else 'Disabled'}")
    logger.info(f"Number of samples to visualize: {args.num_vis_samples}")
    logger.info(f"Show orientation arrows: {'Enabled' if args.show_ori_arrows else 'Disabled'}")
    logger.info(f"Visualize bounding boxes: {'Enabled' if args.visualize_bbox else 'Disabled'}")
    
    # Log Rerun visualization settings
    logger.info("\n=== RERUN VISUALIZATION SETTINGS ===")
    if args.use_rerun:
        if HAS_RERUN:
            logger.info("Rerun visualization for saving recordings: Enabled (viewer will not spawn)")
            logger.info(f"Rerun output directory: {os.path.join(output_dir, 'rerun_visualization')}")
            logger.info(f"Point cloud downsample factor: {args.pointcloud_downsample_factor}")
            logger.info(f"Line width: {args.rerun_line_width}")
            logger.info(f"Point size: {args.rerun_point_size}")
        else:
            logger.info("Rerun visualization: Requested but not available")
            args.use_rerun = False  # Disable Rerun if not available
    else:
        logger.info("Rerun visualization: Disabled")
    logger.info("=============================\n")
    
    # Log command line arguments
    logger.info("Command line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Load model and config
    try:
        model, config, best_epoch = load_model_and_config(args.model_path, device, logger)
        
        # Override config with command line arguments if specified
        if args.use_first_frame_only:
            logger.info(f"Overriding config: use_first_frame_only set to True (from command line)")
            config.use_first_frame_only = True
            
        # Note: show_ori_arrows will be used directly from args in the evaluate function
            
        # --- Print Model Architecture Details ---
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("\n=== MODEL ARCHITECTURE ===")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Text Embedding Enabled: {not getattr(config, 'no_text_embedding', False)}")
        logger.info(f"Use First Frame Only: {config.use_first_frame_only}")
        
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
            logger.info(f"\n{name}:")
            logger.info(f"  Parameters: {params:,}")
            try:
                # Try to log attributes if available
                for attr in attrs:
                    if hasattr(component, attr):
                        logger.info(f"  {attr}: {getattr(component, attr)}")
            except:
                pass
        
        logger.info("=== END MODEL ARCHITECTURE ===\n")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Run evaluation
    evaluate(model, config, args, best_epoch, logger, 
             best_rerun_dir, best_vis_dir, worst_rerun_dir, worst_vis_dir)
    logger.info("Evaluation completed.")

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