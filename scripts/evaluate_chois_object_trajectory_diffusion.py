#!/usr/bin/env python3
"""
Evaluation script for CHOIS Object Trajectory Diffusion Model.

This script evaluates the ObjectTrajectoryDiffusion model that:
1. Processes object trajectories (12D: xyz + 9D rotation matrix)
2. Uses sparse conditioning (initial pose + final position + xy waypoints)
3. Conditions on scene bounding boxes and text categories
4. Based on CHOIS architecture but simplified for trajectory generation only
"""

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
from functools import partial
import shutil

# --- Imports from our project ---
from model.chois_object_trajectory_diffusion_model import ObjectTrajectoryDiffusion
from dataset.gimo_multi_sequence_dataset import GIMOMultiSequenceDataset
from config.chois_config import CHOISObjectTrajectoryConfig
from utils.visualization import visualize_trajectory, visualize_prediction, visualize_full_trajectory
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

# Import ADT Sequence Utilities
try:
    from ariaworldgaussians.adt_sequence_utils import find_adt_sequences, create_train_test_split
    HAS_SEQ_UTILS = True
except ImportError:
    print("Warning: Could not import adt_sequence_utils.")
    HAS_SEQ_UTILS = False

# Import CLIP for text encoding
import clip

# Set random seed
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def extract_pointcloud_along_trajectory(scene_pc: torch.Tensor, trajectory: torch.Tensor, radius: float) -> Optional[torch.Tensor]:
    """
    Extracts a point cloud from the scene that is within a certain radius of any point on the trajectory.
    This is intended for visualization purposes during evaluation.

    Args:
        scene_pc (torch.Tensor): The full scene point cloud, shape [N, 3].
        trajectory (torch.Tensor): The object trajectory, shape [T, 3].
        radius (float): The radius around the trajectory to include points.

    Returns:
        Optional[torch.Tensor]: The extracted point cloud, shape [M, 3], or None if inputs are invalid.
    """
    if scene_pc is None or trajectory is None or scene_pc.numel() == 0 or trajectory.numel() == 0:
        return None
    
    # Ensure tensors are on the same device
    device = trajectory.device
    scene_pc = scene_pc.to(device)
    
    # Calculate squared distances for efficiency
    radius_sq = radius ** 2
    
    # Expand dims for broadcasting: [T, 1, 3] and [1, N, 3]
    trajectory_expanded = trajectory.unsqueeze(1)
    scene_pc_expanded = scene_pc.unsqueeze(0)
    
    # Calculate squared distances between each trajectory point and each scene point
    # dist_sq shape: [T, N]
    dist_sq = torch.sum((trajectory_expanded - scene_pc_expanded) ** 2, dim=2)
    
    # Find the minimum distance from each scene point to any point on the trajectory
    # min_dist_sq shape: [N]
    min_dist_sq, _ = torch.min(dist_sq, dim=0)
    
    # Create a mask for points within the radius
    mask = min_dist_sq <= radius_sq
    
    # Return the filtered point cloud
    return scene_pc[mask]


def setup_logger(output_dir):
    """Set up logger to both write to console and to file."""
    # Create a logger
    logger = logging.getLogger('chois_evaluation')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    log_file = os.path.join(output_dir, 'chois_evaluation.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log start information
    logger.info("=" * 50)
    logger.info(f"CHOIS Object Trajectory Diffusion Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    return logger 


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CHOIS Object Trajectory Diffusion model")
    
    # --- Model and Checkpoint ---
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained ObjectTrajectoryDiffusion checkpoint (.pth file)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_chois_obj_traj",
        help="Directory to save evaluation results"
    )
    
    # --- Dataset Options ---
    parser.add_argument(
        "--adt_dataroot",
        type=str,
        default=None,
        help="Override path to ADT dataroot (directory containing sequences/split files)"
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default=None,
        help="Override path to test split file"
    )
    
    # --- Evaluation Options ---
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    
    # --- Visualization Options ---
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Enable visualization of trajectories"
    )
    parser.add_argument(
        "--num_vis_samples",
        type=int,
        default=500,
        help="Number of trajectory samples to visualize"
    )
    parser.add_argument(
        "--visualize_bbox",
        action="store_true",
        default=True,
        help="Enable visualization of bounding boxes"
    )
    
    # --- Rerun Visualization Options ---
    parser.add_argument(
        "--use_rerun",
        action="store_true",
        default=True,
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
    parser.add_argument(
        "--rerun_show_arrows",
        action="store_true",
        default=True,
        help="Show orientation arrows in Rerun visualization"
    )
    parser.add_argument(
        "--rerun_show_semantic_bboxes", 
        action="store_true",
        default=True,
        help="Show semantic bounding boxes in Rerun visualization"
    )
    parser.add_argument(
        "--rerun_arrow_length",
        type=float,
        default=0.2,
        help="Length of orientation arrows in Rerun visualization"
    )
    
    # --- Point Cloud Options ---
    parser.add_argument(
        "--use_full_scene_pointcloud",
        action="store_true",
        help="Use complete scene pointcloud for visualization instead of trajectory-specific pointcloud"
    )
    parser.add_argument(
        '--use_trajectory_pointcloud',
        type=float,
        default=1.0,
        metavar='RADIUS',
        help='Extract and use a trajectory-specific point cloud for visualization with the specified radius.'
    )
    
    # --- CHOIS Model Configuration Overrides ---
    parser.add_argument(
        "--conditioning_strategy",
        type=str,
        default=None,
        choices=['full_trajectory', 'history_fraction', 'chois_original'],
        help="Override conditioning strategy for evaluation"
    )
    parser.add_argument(
        "--history_fraction",
        type=float,
        default=None,
        help="Override history fraction for evaluation"
    )
    parser.add_argument(
        "--waypoint_interval",
        type=int,
        default=None,
        help="Override waypoint interval for chois_original strategy"
    )
    parser.add_argument(
        "--use_single_step_prediction",
        action="store_true",
        default=True,
        help="Use single-step prediction instead of full diffusion sampling"
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
        default=4,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--global_cache_dir",
        type=str,
        default=None,
        help="Path to a shared global directory for trajectory cache"
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
    Load ObjectTrajectoryDiffusion model and its config from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pth)
        device: Device to load the model on
        logger: Logger instance for logging information
        
    Returns:
        model: Loaded ObjectTrajectoryDiffusion model
        config: Configuration object from the checkpoint
        best_epoch (int): The epoch number at which this checkpoint was saved
    """
    logger.info(f"Loading CHOIS ObjectTrajectoryDiffusion checkpoint from {checkpoint_path}")
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
    config = argparse.Namespace(**config_dict)
    
    logger.info("Configuration loaded from checkpoint:")
    logger.info(str(config))

    # Create ObjectTrajectoryDiffusion model instance using loaded config
    logger.info("Creating ObjectTrajectoryDiffusion model...")
    model = ObjectTrajectoryDiffusion(
        d_feats=config.d_feats,
        d_model=config.d_model,
        n_head=config.n_head,
        n_dec_layers=config.n_dec_layers,
        d_k=config.d_k,
        d_v=config.d_v,
        max_timesteps=config.max_timesteps,
        timesteps=config.diffusion_timesteps,
        loss_type=config.diffusion_loss_type,
        objective=config.objective,
        beta_schedule=config.beta_schedule,
        p2_loss_weight_gamma=config.p2_loss_weight_gamma,
        p2_loss_weight_k=config.p2_loss_weight_k,
        use_bps=getattr(config, 'use_bps', False),
        bps_input_dim=getattr(config, 'bps_input_dim', 3072),
        bps_hidden_dim=getattr(config, 'bps_hidden_dim', 512),
        bps_output_dim=getattr(config, 'bps_output_dim', 256),
        bps_num_points=getattr(config, 'bps_num_points', 1024),
        use_text_embedding=getattr(config, 'text_embedding', False)
    ).to(device)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    param_count_millions = param_count / 1_000_000
    logger.info(f"Loaded ObjectTrajectoryDiffusion model with {param_count_millions:.2f}M parameters.")
    
    # Get the epoch number
    best_epoch = checkpoint.get('epoch', 0)
    if best_epoch > 0:
        logger.info(f"Checkpoint was saved at epoch: {best_epoch}")
    else:
        logger.warning("Warning: Epoch number not found in checkpoint.")
        
    return model, config, best_epoch 


def prepare_batch_for_object_diffusion(batch, config, device):
    """
    Prepare batch data for simplified ObjectTrajectoryDiffusion model.
    Following original CHOIS design - much simpler now.
    """
    # Extract trajectory data (9D: xyz + 6D rotation) - model will convert internally
    poses_9d = batch['full_poses'].float().to(device)  # BS X T X 9
    attention_mask = batch['full_attention_mask'].float().to(device)  # BS X T
    
    # Extract object categories for text encoding
    object_categories = batch.get('object_category', None)
    
    # Extract bbox_corners for BPS computation
    bbox_corners = None
    if 'bbox_corners' in batch and batch['bbox_corners'] is not None:
        bbox_corners = batch['bbox_corners'].float().to(device)  # BS X T X 8 X 3
    else:
        print("Warning: No bbox_corners found in batch, will use dummy BPS")
    
    # Prepare the simplified batch dictionary for our model
    model_batch = {
        'poses': poses_9d,  # Keep as 9D, model will convert internally
        'attention_mask': attention_mask,
        'object_category': object_categories,
        'bbox_corners': bbox_corners,  # Add bbox_corners for BPS computation
    }
    
    # Add original batch data for metadata
    for key in ['object_name', 'sequence_path', 'segment_idx']:
        if key in batch:
            model_batch[key] = batch[key]
    
    return model_batch


def encode_text_categories(categories, clip_model, device):
    """
    Encode object categories using CLIP.
    
    Args:
        categories: List of category strings
        clip_model: CLIP model
        device: PyTorch device
        
    Returns:
        Text features: BS X 512
    """
    if categories is None:
        return None
    
    try:
        # Tokenize and encode text
        text_tokens = clip.tokenize(categories, truncate=True).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens).float()
        return text_features
    except Exception as e:
        print(f"Warning: CLIP encoding failed: {e}")
        return None


def denormalize_trajectory(normalized_trajectory, normalization_params):
    """Denormalize a trajectory using provided parameters."""
    if not normalization_params['is_normalized']:
        return normalized_trajectory

    # Ensure parameters are tensors on the correct device
    scene_min_val = normalization_params.get('scene_min')
    scene_max_val = normalization_params.get('scene_max')
    scene_scale_val = normalization_params.get('scene_scale')

    if not all(isinstance(p, torch.Tensor) for p in [scene_min_val, scene_max_val, scene_scale_val]):
         print("Warning: Invalid normalization parameters found during denormalization. Returning original trajectory.")
         return normalized_trajectory

    scene_min = scene_min_val.to(normalized_trajectory.device)
    scene_max = scene_max_val.to(normalized_trajectory.device)
    scene_scale = scene_scale_val.to(normalized_trajectory.device)

    # Denormalize: [-1, 1] -> [0, 1] -> world
    denormalized = (normalized_trajectory + 1.0) / 2.0
    denormalized = denormalized * scene_scale + scene_min
    return denormalized 


def evaluate(model, config, args, best_epoch, logger):
    """
    Run the evaluation loop for CHOIS ObjectTrajectoryDiffusion model.
    
    Args:
        model: The loaded ObjectTrajectoryDiffusion model
        config: The configuration object (loaded from checkpoint)
        args: Command line arguments for evaluation
        best_epoch (int): The epoch number the model checkpoint was saved at
        logger: Logger instance for logging information
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Apply command line overrides to config
    if args.conditioning_strategy is not None:
        config.conditioning_strategy = args.conditioning_strategy
        logger.info(f"Override: conditioning_strategy = {args.conditioning_strategy}")
    if args.history_fraction is not None:
        config.history_fraction = args.history_fraction
        logger.info(f"Override: history_fraction = {args.history_fraction}")
    if args.waypoint_interval is not None:
        config.waypoint_interval = args.waypoint_interval
        logger.info(f"Override: waypoint_interval = {args.waypoint_interval}")

    # --- Setup Dataset and DataLoader ---
    logger.info("Setting up test dataset...")
    test_sequences = None
    checkpoint_dir = os.path.dirname(args.model_path)
    val_split_path = os.path.join(checkpoint_dir, 'val_sequences.txt')
    dataroot_override = args.adt_dataroot

    if os.path.exists(val_split_path):
        try:
            logger.info(f"Loading test sequences from saved validation split: {val_split_path}")
            with open(val_split_path, 'r') as f:
                test_sequences = [line.strip() for line in f if line.strip()]
            if not test_sequences:
                logger.warning(f"Warning: {val_split_path} is empty.")
                test_sequences = None
            else:
                logger.info(f"Loaded {len(test_sequences)} test sequences.")
                if dataroot_override:
                    adjusted_sequences = []
                    for seq in test_sequences:
                        if not os.path.isabs(seq) and not seq.startswith(dataroot_override):
                            base_name = os.path.basename(seq)
                            adjusted_path = os.path.join(dataroot_override, base_name)
                            if os.path.exists(adjusted_path):
                                adjusted_sequences.append(adjusted_path)
                            else:
                                logger.warning(f"Warning: Could not adjust path for {seq} using override {dataroot_override}")
                                adjusted_sequences.append(seq)
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
            logger.warning(f"Warning: Using ALL sequences found in {dataroot} as test set due to missing split info.")
            try:
                if HAS_SEQ_UTILS:
                    from ariaworldgaussians.adt_sequence_utils import find_adt_sequences
                    test_sequences = find_adt_sequences(dataroot)
                else:
                    test_sequences = [os.path.join(dataroot, item) for item in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, item))]
            except Exception as e:
                logger.error(f"Error scanning dataroot: {e}")
                return None

            if not test_sequences:
                logger.error(f"Error: No sequences found in fallback dataroot {dataroot}.")
                return None
            logger.info(f"Found {len(test_sequences)} sequences in fallback dataroot.")
        elif os.path.exists(dataroot):
            test_sequences = [dataroot]
            logger.info(f"Using single sequence {dataroot} as test set (fallback).")
        else:
            logger.error(f"Error: Fallback failed. Dataroot path {dataroot} not found or invalid.")
            return None

    # Use cache dir based on output dir
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
        config=config,
        cache_dir=eval_cache_dir,
        use_cache=True
    )

    if len(test_dataset) == 0:
        logger.error("Error: Test dataset is empty. Check test split file and sequence paths.")
        return None

    # Use the collate function, binding the test dataset instance
    eval_collate_func = partial(gimo_collate_fn, dataset=test_dataset, num_sample_points=config.sample_points)
    eval_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=eval_collate_func
    )
    logger.info(f"Test DataLoader ready with {len(test_dataset)} samples.")

    # Initialize CLIP for text encoding if needed
    clip_model = None
    if getattr(config, 'text_embedding', False):
        try:
            clip_model, _ = clip.load("ViT-B/32", device=device)
            clip_model.eval()
            for param in clip_model.parameters():
                param.requires_grad = False
            logger.info("CLIP model loaded for text encoding")
        except Exception as e:
            logger.warning(f"Warning: Could not load CLIP model: {e}")

    # --- Evaluation Loop ---
    total_l1 = 0.0
    total_rmse_ade = 0.0
    total_fde = 0.0
    total_frechet = 0.0
    total_angular_cosine = 0.0
    num_batches = 0
    total_valid_samples = 0
    all_samples_metrics = []

    # Track visualization statistics
    visualized_count = 0
    skipped_visualizations = 0

    # For Matplotlib visualization
    vis_output_dir = os.path.join(args.output_dir, "visualizations")
    if args.visualize:
        os.makedirs(vis_output_dir, exist_ok=True)
        logger.info(f"Matplotlib visualizations will be saved to {vis_output_dir}")

    # Base directory for per-sample Rerun .rrd files
    per_sample_rrd_basedir = None
    if args.use_rerun and HAS_RERUN:
        per_sample_rrd_basedir = os.path.join(args.output_dir, "rerun_visualizations_per_sample")
        os.makedirs(per_sample_rrd_basedir, exist_ok=True)
        logger.info(f"Rerun per-sample .rrd files will be saved to: {per_sample_rrd_basedir}")

    inference_method = "Single-step prediction" if args.use_single_step_prediction else "Diffusion sampling"
    logger.info(f"Using {inference_method} for trajectory generation")

    with torch.no_grad():
        progress_bar = tqdm(eval_loader, desc="Evaluating")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                logger.info(f"Processing batch {batch_idx+1}/{len(eval_loader)}")
                
                # Prepare batch for our model
                model_batch = prepare_batch_for_object_diffusion(batch, config, device)
                
                # Encode text features if available
                text_features = None
                if model_batch['object_category'] is not None and clip_model is not None:
                    text_features = encode_text_categories(
                        model_batch['object_category'], clip_model, device
                    )
                model_batch['text_features'] = text_features
                
                # Forward pass for loss computation
                model_out, val_loss = model(
                    model_batch, 
                    history_fraction=config.history_fraction,
                    conditioning_strategy=getattr(config, 'conditioning_strategy', 'history_fraction'),
                    waypoint_interval=getattr(config, 'waypoint_interval', 30)
                )
                
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                # Generate trajectories for metrics and visualization
                if args.use_single_step_prediction:
                    generated_trajectories_12d = model_out
                else:
                    logger.warning("Diffusion sampling not implemented for CHOIS model, using single-step prediction")
                    generated_trajectories_12d = model_out
                
                # Convert back to 9D for metrics calculation
                generated_trajectories_9d = model.convert_12d_to_9d(generated_trajectories_12d)
                
                # Get normalization params
                normalization_params = batch.get('normalization', {'is_normalized': False})
                object_names = batch.get('object_name', [])
                segment_indices = batch.get('segment_idx', [])
                
                # Process Each Sample in Batch
                batch_l1 = 0.0
                batch_rmse_ade = 0.0
                batch_fde = 0.0
                batch_frechet = 0.0
                batch_angular_cosine = 0.0
                valid_samples_in_batch = 0

                logger.info(f"Processing {model_batch['poses'].shape[0]} samples in batch")
                for i in range(model_batch['poses'].shape[0]):
                    # Extract ground truth data
                    gt_poses_9d = model_batch['poses'][i]  # 9D
                    gt_mask = model_batch['attention_mask'][i]
                    pred_poses_12d = generated_trajectories_12d[i]  # 12D
                    
                    # Get actual length from mask
                    actual_length = torch.sum(gt_mask).int().item()
                    if actual_length < 2:
                        logger.info(f"Sample {i} skipped: trajectory too short (length={actual_length})")
                        continue
                    
                    # Determine history length based on config
                    history_length = max(1, int(actual_length * config.history_fraction))
                    history_length = min(history_length, actual_length - 1)
                    
                    # Extract position component for metrics
                    gt_positions = gt_poses_9d[:, :3]  # Extract xyz
                    pred_positions = pred_poses_12d[:, :3]  # Extract xyz
                    
                    # Split into history and future
                    gt_future_positions = gt_positions[history_length:actual_length]
                    pred_future_positions = pred_positions[history_length:actual_length]
                    future_mask = gt_mask[history_length:actual_length]
                    
                    if future_mask.sum() == 0:
                        continue
                    
                    # Extract normalization params for this specific sample
                    sample_norm_params = normalization_params
                    if isinstance(normalization_params, list) and i < len(normalization_params):
                        sample_norm_params = normalization_params[i]
                    
                    # Denormalize if necessary
                    is_normalized = sample_norm_params.get('is_normalized', False)
                    if isinstance(is_normalized, torch.Tensor):
                        is_normalized = is_normalized.item()
                    
                    if is_normalized:
                        gt_future_denorm = denormalize_trajectory(gt_future_positions, sample_norm_params)
                        pred_future_denorm = denormalize_trajectory(pred_future_positions, sample_norm_params)
                        gt_hist_denorm = denormalize_trajectory(gt_positions[:history_length], sample_norm_params)
                    else:
                        gt_future_denorm = gt_future_positions
                        pred_future_denorm = pred_future_positions
                        gt_hist_denorm = gt_positions[:history_length]
                    
                    # Compute Metrics for this sample
                    l1_mean, rmse_ade, fde, frechet_distance, angular_cosine_similarity = compute_metrics_for_sample(
                        pred_future_denorm, gt_future_denorm, future_mask
                    )
                    
                    logger.info(f"Sample {i}: L1={l1_mean.item():.4f}, RMSE={rmse_ade.item():.4f}, FDE={fde.item():.4f}")
                    
                    batch_l1 += l1_mean.item()
                    batch_rmse_ade += rmse_ade.item()
                    batch_fde += fde.item()
                    batch_frechet += frechet_distance.item()
                    batch_angular_cosine += angular_cosine_similarity.item()
                    valid_samples_in_batch += 1
                    
                    # Store sample metrics for later sorting
                    all_samples_metrics.append({
                        'l1': l1_mean.item(),
                        'rmse': rmse_ade.item(),
                        'fde': fde.item(),
                        'frechet': frechet_distance.item(),
                        'angular_cosine': angular_cosine_similarity.item(),
                        'val_loss': val_loss.item(),
                        'vis_path': None,  # Will be set during visualization
                        'rerun_path': None,  # Will be set during visualization
                        'sample_name': f"batch{batch_idx}_sample{i}"
                    })
                    
                    # Visualization
                    if args.visualize and visualized_count < args.num_vis_samples:
                        try:
                            # Create visualization paths and metadata
                            obj_name = object_names[i] if i < len(object_names) else f"obj_{batch_idx}_{i}"
                            seg_idx = segment_indices[i].item() if i < len(segment_indices) and segment_indices[i].item() != -1 else None
                            
                            current_sequence_path = batch['sequence_path'][i]
                            sequence_name_extracted = os.path.splitext(os.path.basename(current_sequence_path))[0]
                            
                            filename_base = f"{obj_name}_seq_{sequence_name_extracted}_seg{seg_idx}_batch{batch_idx}"
                            vis_title_base = f"{obj_name} (Seq: {sequence_name_extracted}, Seg: {seg_idx}, Batch: {batch_idx})"
                            
                            logger.info(f"Visualizing sample {i}: {vis_title_base}")
                            
                            # Apply coordinate transformation for visualization
                            gt_hist_vis = transform_coords_for_visualization(gt_hist_denorm.cpu())
                            gt_future_vis = transform_coords_for_visualization(gt_future_denorm.cpu())
                            pred_future_vis = transform_coords_for_visualization(pred_future_denorm.cpu())
                            
                            # Standard prediction vs ground truth visualization
                            pred_vs_gt_path = os.path.join(vis_output_dir, f"{filename_base}_prediction_vs_gt.png")
                            
                            visualize_prediction(
                                past_positions=gt_hist_vis,
                                future_positions_gt=gt_future_vis,
                                future_positions_pred=pred_future_vis,
                                past_mask=gt_mask[:history_length].cpu(),
                                future_mask_gt=future_mask.cpu(),
                                title=f"CHOIS Pred vs GT - {vis_title_base}",
                                save_path=pred_vs_gt_path,
                                segment_idx=seg_idx,
                                show_orientation=args.show_ori_arrows
                            )
                            
                            # Add full trajectory visualization with bounding boxes if requested
                            if args.visualize_bbox and 'bbox_corners' in batch:
                                gt_full_denorm = torch.cat([gt_hist_denorm, gt_future_denorm], dim=0)[:actual_length]
                                full_mask = torch.cat([gt_mask[:history_length], future_mask], dim=0)[:actual_length]
                                
                                full_traj_path = os.path.join(vis_output_dir, f"{filename_base}_full_trajectory_with_bbox.png")
                                
                                # Get the sample's bounding box corners
                                sample_bbox_corners_cpu = batch['bbox_corners'][i, :actual_length].cpu()
                                
                                # Denormalize bounding box corners if necessary
                                if is_normalized:
                                    bbox_shape = sample_bbox_corners_cpu.shape
                                    sample_bbox_corners_flat = sample_bbox_corners_cpu.reshape(-1, 3)
                                    sample_bbox_corners_flat_denorm = denormalize_trajectory(
                                        sample_bbox_corners_flat, 
                                        sample_norm_params
                                    )
                                    sample_bbox_corners_cpu = sample_bbox_corners_flat_denorm.reshape(bbox_shape)
                                
                                # Apply transformation for visualization
                                gt_full_denorm_vis = transform_coords_for_visualization(gt_full_denorm.cpu())
                                
                                # Choose point cloud for visualization based on user preference
                                sample_point_cloud_vis = None
                                
                                if args.use_trajectory_pointcloud is not None:
                                    # Priority 1: Extract point cloud along the trajectory from the scene_pointcloud
                                    if 'scene_pointcloud' in batch and i < len(batch['scene_pointcloud']):
                                        scene_pc_data = batch['scene_pointcloud'][i]
                                        if scene_pc_data is not None and isinstance(scene_pc_data, torch.Tensor) and scene_pc_data.numel() > 0:
                                            extracted_pc = extract_pointcloud_along_trajectory(
                                                scene_pc=scene_pc_data.cpu(),
                                                trajectory=gt_full_denorm.cpu(),
                                                radius=args.use_trajectory_pointcloud
                                            )
                                            if extracted_pc is not None:
                                                sample_point_cloud_vis = transform_coords_for_visualization(extracted_pc)
                                                logger.info(f"Using trajectory-specific pointcloud (radius={args.use_trajectory_pointcloud}) with {extracted_pc.shape[0]} points.")
                                        else:
                                            logger.warning("`--use_trajectory_pointcloud` was specified but scene_pointcloud is not available for this sample.")
                                
                                elif args.use_full_scene_pointcloud:
                                    # Priority 2: Use complete scene point cloud if available and requested
                                    if 'scene_pointcloud' in batch and i < len(batch['scene_pointcloud']):
                                        scene_pc_data = batch['scene_pointcloud'][i]
                                        if scene_pc_data is not None and isinstance(scene_pc_data, torch.Tensor) and scene_pc_data.numel() > 0:
                                            sample_point_cloud_vis = transform_coords_for_visualization(scene_pc_data.cpu())
                                            logger.info(f"Using full scene pointcloud for visualization ({scene_pc_data.shape[0]} points)")
                                
                                if sample_point_cloud_vis is None:
                                    # Fallback to the default point cloud from the batch
                                    if 'point_cloud' in batch:
                                        sample_point_cloud_vis = transform_coords_for_visualization(batch['point_cloud'][i].cpu())
                                        logger.info(f"Using default pointcloud from dataset for visualization.")
                                
                                sample_bbox_corners_vis = transform_coords_for_visualization(sample_bbox_corners_cpu)
                                
                                # Visualize the full trajectory with bbox
                                visualize_full_trajectory(
                                    positions=gt_full_denorm_vis,
                                    attention_mask=full_mask,
                                    point_cloud=sample_point_cloud_vis,
                                    bbox_corners_sequence=sample_bbox_corners_vis,
                                    title=f"Full CHOIS Trajectory - {vis_title_base}",
                                    save_path=full_traj_path,
                                    segment_idx=seg_idx
                                )
                            
                            # --- Rerun Visualization (per sample) ---
                            if args.use_rerun and HAS_RERUN and per_sample_rrd_basedir:
                                try:
                                    # Construct unique names and paths for this sample's Rerun recording
                                    sample_recording_name = filename_base
                                    logger.info(f"Attempting Rerun visualization for sample: {sample_recording_name}")

                                    # Initialize Rerun for THIS SPECIFIC SAMPLE
                                    sample_rerun_initialized, returned_sample_rrd_path = initialize_rerun(
                                        recording_name=sample_recording_name, 
                                        spawn=False,  # No viewer per sample
                                        output_dir=per_sample_rrd_basedir 
                                    )

                                    if sample_rerun_initialized:
                                        # --- Prepare point cloud for Rerun ---
                                        traj_point_cloud = None
                                        
                                        if args.use_trajectory_pointcloud is not None:
                                            # Priority 1 for Rerun: Extract from scene_pointcloud
                                            if 'scene_pointcloud' in batch and i < len(batch['scene_pointcloud']):
                                                scene_pc_data = batch['scene_pointcloud'][i]
                                                if scene_pc_data is not None and isinstance(scene_pc_data, torch.Tensor) and scene_pc_data.numel() > 0:
                                                    # Create full trajectory for point cloud extraction
                                                    gt_full_denorm_for_pc = torch.cat([gt_hist_denorm, gt_future_denorm], dim=0)[:actual_length]
                                                    traj_point_cloud = extract_pointcloud_along_trajectory(
                                                        scene_pc=scene_pc_data.cpu(),
                                                        trajectory=gt_full_denorm_for_pc.cpu(),
                                                        radius=args.use_trajectory_pointcloud
                                                    )
                                        
                                        elif args.use_full_scene_pointcloud:
                                            # Priority 2 for Rerun: Use full scene pointcloud
                                            if 'scene_pointcloud' in batch and i < len(batch['scene_pointcloud']):
                                                scene_pc_data = batch['scene_pointcloud'][i]
                                                if scene_pc_data is not None and isinstance(scene_pc_data, torch.Tensor) and scene_pc_data.numel() > 0:
                                                    traj_point_cloud = scene_pc_data.cpu()
                                        
                                        if traj_point_cloud is None:
                                            # Fallback for Rerun: Use default point cloud (from dataset)
                                            if 'point_cloud' in batch and batch['point_cloud'].shape[0] > i:
                                                traj_point_cloud = batch['point_cloud'][i].cpu()

                                        # Downsample the selected point cloud
                                        if traj_point_cloud is not None and args.pointcloud_downsample_factor > 1:
                                            traj_point_cloud = downsample_point_cloud(
                                                traj_point_cloud, 
                                                args.pointcloud_downsample_factor
                                            )
                                        
                                        # Apply transformation for Rerun visualization
                                        gt_hist_denorm_rerun_vis = transform_coords_for_visualization(gt_hist_denorm.cpu())
                                        gt_future_denorm_rerun_vis = transform_coords_for_visualization(gt_future_denorm.cpu())
                                        pred_future_denorm_rerun_vis = transform_coords_for_visualization(pred_future_denorm.cpu())
                                        traj_point_cloud_rerun_vis = transform_coords_for_visualization(traj_point_cloud.cpu() if traj_point_cloud is not None else None)
                                        
                                        # Extract orientation data for Rerun
                                        gt_hist_rot_denorm = gt_poses_9d[:history_length, 3:]  # Rotation part
                                        gt_future_rot_denorm = gt_poses_9d[history_length:actual_length, 3:]
                                        pred_future_rot_denorm = pred_poses_12d[history_length:actual_length, 3:12]  # 9D rotation from 12D prediction
                                        
                                        # Log data for this sample with enhanced visualization
                                        visualize_trajectory_rerun(
                                            past_positions=gt_hist_denorm_rerun_vis,
                                            future_positions_gt=gt_future_denorm_rerun_vis,
                                            future_positions_pred=pred_future_denorm_rerun_vis,
                                            past_mask=gt_mask[:history_length],
                                            future_mask_gt=future_mask,
                                            point_cloud=traj_point_cloud_rerun_vis,
                                            past_orientations=gt_hist_rot_denorm.cpu(),
                                            future_orientations_gt=gt_future_rot_denorm.cpu(),
                                            future_orientations_pred=pred_future_rot_denorm.cpu(),
                                            semantic_bbox_info=None,  # CHOIS doesn't use semantic bboxes
                                            semantic_bbox_mask=None,
                                            semantic_bbox_categories=None,
                                            object_name=obj_name, 
                                            sequence_name=sequence_name_extracted, 
                                            segment_idx=seg_idx, 
                                            arrow_length=args.rerun_arrow_length,
                                            line_width=args.rerun_line_width,
                                            point_size=args.rerun_point_size,
                                            show_arrows=args.rerun_show_arrows,
                                            show_semantic_bboxes=False  # CHOIS doesn't use semantic bboxes
                                        )

                                        # Explicitly save if init fell back
                                        if returned_sample_rrd_path:
                                            logger.info(f"Attempting to explicitly save/finalize Rerun recording for sample {sample_recording_name} to: {returned_sample_rrd_path}")
                                            save_rerun_recording(output_path=returned_sample_rrd_path)
                                        
                                        # Disconnect Rerun session for this sample
                                        if hasattr(rr, 'disconnect') and callable(rr.disconnect):
                                            try:
                                                rr.disconnect()
                                                logger.info(f"Disconnected Rerun session for sample {sample_recording_name}")
                                            except Exception as disconnect_e:
                                                logger.warning(f"Error during Rerun disconnect for sample {sample_recording_name}: {disconnect_e}")

                                    else:
                                        logger.warning(f"Failed to initialize Rerun for sample: {sample_recording_name}. Skipping Rerun vis for this sample.")
                                        
                                    # Update metrics with Rerun path
                                    if returned_sample_rrd_path and os.path.exists(returned_sample_rrd_path):
                                        all_samples_metrics[-1]['rerun_path'] = returned_sample_rrd_path
                                    
                                except Exception as rerun_sample_e:
                                    logger.warning(f"Warning: Error during Rerun visualization for sample {i} in batch {batch_idx} ({filename_base}): {rerun_sample_e}")
                            
                            # Update metrics with visualization path
                            all_samples_metrics[-1]['vis_path'] = pred_vs_gt_path
                            all_samples_metrics[-1]['sample_name'] = filename_base
                            
                            visualized_count += 1
                            
                        except Exception as vis_e:
                            logger.warning(f"Warning: Error during visualization for sample {i} in batch {batch_idx}: {vis_e}")

                # Aggregate Batch Metrics
                if valid_samples_in_batch > 0:
                    total_l1 += batch_l1
                    total_rmse_ade += batch_rmse_ade
                    total_fde += batch_fde
                    total_frechet += batch_frechet
                    total_angular_cosine += batch_angular_cosine
                    total_valid_samples += valid_samples_in_batch
                    num_batches += 1
                    
                    batch_avg_l1 = batch_l1 / valid_samples_in_batch
                    batch_avg_rmse = batch_rmse_ade / valid_samples_in_batch
                    batch_avg_fde = batch_fde / valid_samples_in_batch
                    
                    logger.info(f"Batch {batch_idx} metrics: L1={batch_avg_l1:.4f}, RMSE={batch_avg_rmse:.4f}, FDE={batch_avg_fde:.4f}")
                    progress_bar.set_postfix({
                        'L1': f"{batch_avg_l1:.4f}",
                        'RMSE': f"{batch_avg_rmse:.4f}",
                        'FDE': f"{batch_avg_fde:.4f}"
                    })

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

    # Calculate Final Metrics
    if total_valid_samples > 0:
        mean_l1 = total_l1 / total_valid_samples
        mean_rmse = total_rmse_ade / total_valid_samples
        mean_fde = total_fde / total_valid_samples
        mean_frechet = total_frechet / total_valid_samples
        mean_angular_cosine = total_angular_cosine / total_valid_samples

        logger.info(f"--- CHOIS Object Trajectory Diffusion Evaluation Complete ---")
        logger.info(f" Model from Epoch: {best_epoch}")
        logger.info(f" Mean L1 (MAE): {mean_l1:.4f}")
        logger.info(f" Mean RMSE: {mean_rmse:.4f}")
        logger.info(f" Mean FDE: {mean_fde:.4f}")
        logger.info(f" Mean Fréchet: {mean_frechet:.4f}")
        logger.info(f" Mean Angular_Cos: {mean_angular_cosine:.4f}")
        logger.info(f" Visualized {visualized_count} samples")
    else:
        logger.error("Error: No batches were successfully processed.")
        mean_l1 = float('inf')
        mean_rmse = float('inf')
        mean_fde = float('inf')
        mean_frechet = float('inf')
        mean_angular_cosine = float('inf')

    # Save Results
    results = {
        'model_path': args.model_path,
        'best_model_epoch': best_epoch,
        'config': vars(config),
        'eval_args': vars(args),
        'mean_l1': mean_l1,
        'mean_rmse': mean_rmse,
        'mean_fde': mean_fde,
        'mean_frechet': mean_frechet,
        'mean_angular_cosine': mean_angular_cosine,
        'num_test_samples_processed': total_valid_samples,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'num_visualized_samples': visualized_count,
        'inference_method': inference_method,
        'conditioning_strategy': getattr(config, 'conditioning_strategy', 'history_fraction'),
        'history_fraction': getattr(config, 'history_fraction', 0.3)
    }

    results_path = os.path.join(args.output_dir, "chois_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Evaluation results saved to {results_path}")

    # --- Sort and Save Best/Worst Reruns and Visualizations by Different Metrics ---
    if all_samples_metrics:
        num_to_save = args.num_top_worst_to_save
        logger.info(f"Saving top {num_to_save} Rerun recordings and visualizations for each metric...")
        
        # Create separate directories for different metrics
        l1_best_rerun_dir = os.path.join(args.output_dir, "l1_best_rerun")
        l1_best_vis_dir = os.path.join(args.output_dir, "l1_best_visualization")
        frechet_best_rerun_dir = os.path.join(args.output_dir, "frechet_best_rerun")
        frechet_best_vis_dir = os.path.join(args.output_dir, "frechet_best_visualization")
        angular_best_rerun_dir = os.path.join(args.output_dir, "angular_cosine_best_rerun")
        angular_best_vis_dir = os.path.join(args.output_dir, "angular_cosine_best_visualization")
        
        os.makedirs(l1_best_rerun_dir, exist_ok=True)
        os.makedirs(l1_best_vis_dir, exist_ok=True)
        os.makedirs(frechet_best_rerun_dir, exist_ok=True)
        os.makedirs(frechet_best_vis_dir, exist_ok=True)
        os.makedirs(angular_best_rerun_dir, exist_ok=True)
        os.makedirs(angular_best_vis_dir, exist_ok=True)
        
        # Define metrics and their directories
        metrics_to_evaluate = [
            {'name': 'L1', 'key': 'l1', 'ascending': True, 'rerun_dir': l1_best_rerun_dir, 'vis_dir': l1_best_vis_dir},
            {'name': 'Fréchet', 'key': 'frechet', 'ascending': True, 'rerun_dir': frechet_best_rerun_dir, 'vis_dir': frechet_best_vis_dir},
            {'name': 'Angular_Cosine', 'key': 'angular_cosine', 'ascending': False, 'rerun_dir': angular_best_rerun_dir, 'vis_dir': angular_best_vis_dir}  # Higher cosine similarity is better
        ]
        
        for metric_info in metrics_to_evaluate:
            metric_name = metric_info['name']
            metric_key = metric_info['key']
            ascending = metric_info['ascending']
            rerun_dir = metric_info['rerun_dir']
            vis_dir = metric_info['vis_dir']
            
            # Sort samples by the current metric
            sorted_samples = sorted(all_samples_metrics, key=lambda x: x[metric_key], reverse=not ascending)
            
            logger.info(f"--- Saving {min(num_to_save, len(sorted_samples))} best samples by {metric_name} ---")
            
            for i in range(min(num_to_save, len(sorted_samples))):
                sample_data = sorted_samples[i]
                logger.info(f"Best {metric_name} sample {i+1}/{num_to_save} ({metric_name}: {sample_data[metric_key]:.4f}): {sample_data['sample_name']}")
                
                # Copy visualization
                if sample_data['vis_path'] and os.path.exists(sample_data['vis_path']):
                    try:
                        shutil.copy(sample_data['vis_path'], vis_dir)
                        logger.info(f"  Copied visualization to {vis_dir}")
                    except Exception as e:
                        logger.error(f"  Error copying visualization {sample_data['vis_path']}: {e}")
                
                # Copy Rerun recording
                if sample_data['rerun_path'] and os.path.exists(sample_data['rerun_path']):
                    try:
                        shutil.copy(sample_data['rerun_path'], rerun_dir)
                        logger.info(f"  Copied Rerun recording to {rerun_dir}")
                    except Exception as e:
                        logger.error(f"  Error copying Rerun recording {sample_data['rerun_path']}: {e}")

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
    
    # Setup logger
    logger = setup_logger(output_dir)
    logger.info(f"Evaluating CHOIS Object Trajectory Diffusion model: {args.model_path}")
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Log evaluation settings
    logger.info("\n=== EVALUATION SETTINGS ===")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Use single step prediction: {args.use_single_step_prediction}")
    logger.info(f"Matplotlib visualization: {'Enabled' if args.visualize else 'Disabled'}")
    logger.info(f"Number of samples to visualize: {args.num_vis_samples}")
    logger.info(f"Show orientation arrows: {'Enabled' if args.show_ori_arrows else 'Disabled'}")
    logger.info(f"Visualize bounding boxes: {'Enabled' if args.visualize_bbox else 'Disabled'}")
    
    if args.use_trajectory_pointcloud:
        logger.info(f"Trajectory-specific point cloud extraction: Enabled (radius={args.use_trajectory_pointcloud})")
    elif args.use_full_scene_pointcloud:
        logger.info("Full scene point cloud: Enabled")
    else:
        logger.info("Default point cloud: Enabled")
    
    logger.info(f"Rerun visualization: {'Enabled' if args.use_rerun and HAS_RERUN else 'Disabled'}")
    logger.info("=============================\n")
    
    # Log command line arguments
    logger.info("Command line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Load model and config
    try:
        model, config, best_epoch = load_model_and_config(args.model_path, device, logger)
        
        # Log model architecture details
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("\n=== CHOIS MODEL ARCHITECTURE DETAILS ===")
        logger.info(f"Model Type: ObjectTrajectoryDiffusion")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        
        # Calculate model size in MB (assuming float32)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        logger.info(f"Estimated Model Size: {model_size_mb:.2f} MB")
        
        # Log configuration flags
        logger.info(f"\n--- Configuration Flags ---")
        logger.info(f"Text Embedding Enabled: {getattr(config, 'text_embedding', False)}")
        logger.info(f"BPS Enabled: {getattr(config, 'use_bps', False)}")
        logger.info(f"Conditioning Strategy: {getattr(config, 'conditioning_strategy', 'history_fraction')}")
        logger.info(f"History Fraction: {getattr(config, 'history_fraction', 0.3)}")
        logger.info(f"Waypoint Interval: {getattr(config, 'waypoint_interval', 30)}")
        logger.info(f"Diffusion Objective: {getattr(config, 'objective', 'pred_x0')}")
        logger.info(f"Diffusion Timesteps: {getattr(config, 'diffusion_timesteps', 1000)}")
        logger.info(f"Beta Schedule: {getattr(config, 'beta_schedule', 'cosine')}")
        
        logger.info("=== END MODEL ARCHITECTURE DETAILS ===\n")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Run evaluation
    results = evaluate(model, config, args, best_epoch, logger)
    if results:
        logger.info("CHOIS Object Trajectory Diffusion evaluation completed successfully.")
    else:
        logger.error("Evaluation failed.")


if __name__ == "__main__":
    main()

# Example usage:
# 1. Basic evaluation with visualization
# python evaluate_chois_object_trajectory_diffusion.py --model_path ./checkpoints/best_obj_traj_model.pth
#
# 2. Evaluation with specific conditioning strategy
# python evaluate_chois_object_trajectory_diffusion.py --model_path ./checkpoints/model.pth --conditioning_strategy chois_original --waypoint_interval 20
#
# 3. Evaluation with trajectory-specific point cloud extraction
# python evaluate_chois_object_trajectory_diffusion.py --model_path ./checkpoints/model.pth --use_trajectory_pointcloud 1.5
#
# 4. Evaluation with custom history fraction
# python evaluate_chois_object_trajectory_diffusion.py --model_path ./checkpoints/model.pth --history_fraction 0.4
#
# 5. Evaluation with Rerun visualization
# python evaluate_chois_object_trajectory_diffusion.py --model_path ./checkpoints/model.pth --use_rerun
#
# 6. Advanced Rerun visualization with custom settings
# python evaluate_chois_object_trajectory_diffusion.py --model_path ./checkpoints/model.pth --use_rerun --rerun_show_arrows --rerun_arrow_length 0.3 --pointcloud_downsample_factor 2
#
# 7. Full evaluation with both matplotlib and rerun, plus bounding box visualization
# python evaluate_chois_object_trajectory_diffusion.py --model_path ./checkpoints/model.pth --visualize --visualize_bbox --use_rerun --show_ori_arrows --rerun_show_arrows
#
# 8. Evaluate with diffusion sampling (if implemented)
# python evaluate_chois_object_trajectory_diffusion.py --model_path ./checkpoints/model.pth --no-use_single_step_prediction
#
# 9. Comprehensive evaluation with custom metrics saving
# python evaluate_chois_object_trajectory_diffusion.py --model_path ./checkpoints/model.pth --use_rerun --num_top_worst_to_save 20 --num_vis_samples 100 