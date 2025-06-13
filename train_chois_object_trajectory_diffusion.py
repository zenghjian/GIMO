#!/usr/bin/env python3
"""
Training script for Object Trajectory Diffusion Model.

This script trains the ObjectTrajectoryDiffusion model that:
1. Processes object trajectories (12D: xyz + 9D rotation matrix)
2. Uses sparse conditioning (initial pose + final position + xy waypoints)
3. Conditions on scene bounding boxes and text categories
4. Based on CHOIS architecture but simplified for trajectory generation only
"""

import torch
import torch.optim as optim
import os
import time
import numpy as np
from tqdm import tqdm
import json
import wandb
from torch.utils.data.dataloader import default_collate
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Imports from our project
from config.chois_config import CHOISObjectTrajectoryConfig
from dataset.gimo_multi_sequence_dataset import GIMOMultiSequenceDataset
from model.chois_object_trajectory_diffusion_model import ObjectTrajectoryDiffusion
from torch.utils.data import DataLoader
from utils.visualization import visualize_trajectory, visualize_prediction, visualize_full_trajectory

# Import metrics utilities
from utils.metrics_utils import (
    transform_coords_for_visualization,
    compute_metrics_for_sample,
    gimo_collate_fn,
    GradientTracker,
    plot_orientation_distribution
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




def log_metrics(epoch, title, metrics, logger_func):
    log_str = f"Epoch {epoch} {title}: Total Loss {metrics['total_loss']:.4f}"
    log_str += " | Components: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'total_loss'])
    logger_func(log_str)


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


def validate_object_diffusion(model, dataloader, device, config, epoch, logger, clip_model=None, use_single_step_prediction=True):
    """
    Validation for Object Trajectory Diffusion model.
    
    Args:
        use_single_step_prediction: If True, use single-step prediction instead of multi-step diffusion sampling.
                                   This is useful when objective='pred_x0' and training loss is low, 
                                   but sampling results are poor. Single-step prediction directly 
                                   evaluates the model's denoising capability without multi-step sampling artifacts.
    """
    model.eval()
    val_total_loss = 0.0
    visualized_count = 0
    vis_limit = config.num_val_visualizations if hasattr(config, 'num_val_visualizations') else 10
    
    # Check if this is the first validation run
    is_first_validation = (epoch == config.val_fre)
    
    # Create output directories
    vis_output_dir = os.path.join(config.save_path, "val_visualizations", f"epoch_{epoch}")
    
    if is_first_validation:
        trajectory_vis_dir = os.path.join(config.save_path, "trajectory_visualizations")
        os.makedirs(trajectory_vis_dir, exist_ok=True)
        print(f"First validation run: trajectory visualizations will be saved to: {trajectory_vis_dir}")
    
    if vis_limit > 0:
        os.makedirs(vis_output_dir, exist_ok=True)
        print(f"Validation visualizations will be saved to: {vis_output_dir}")

    # Metrics tracking
    total_l1 = 0.0
    total_rmse = 0.0
    total_fde = 0.0
    total_frechet = 0.0
    total_angular_cosine = 0.0
    total_valid_samples = 0

    inference_method = "Single-step prediction" if use_single_step_prediction else "Diffusion sampling"
    debug_info = "(DEBUG: FIRST BATCH ONLY)" if config.first_batch else ""
    print(f"\nRunning Object Trajectory Diffusion validation using {inference_method} {debug_info}...")
    logger(f"Validation inference method: {inference_method}")
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare batch for our model
                model_batch = prepare_batch_for_object_diffusion(batch, config, device)
                
                # Encode text features if available
                text_features = None
                if model_batch['object_category'] is not None and clip_model is not None:
                    text_features = encode_text_categories(
                        model_batch['object_category'], clip_model, device
                    )
                model_batch['text_features'] = text_features
                
                # Forward pass (validation)
                val_model_out, val_loss = model(model_batch, history_fraction=config.history_fraction, conditioning_strategy=config.conditioning_strategy, waypoint_interval=config.waypoint_interval)
                
                val_total_loss += val_loss.item()
                
                progress_bar.set_postfix({'val_loss': f"{val_loss.item():.4f}"})
                
                # Generate trajectories for metrics and visualization
                if visualized_count < vis_limit:
                    try:
                        sequence_length = model_batch['poses'].shape[1]
                        batch_size = model_batch['poses'].shape[0]
                        
                        if use_single_step_prediction:
                            # Single-step prediction: directly predict clean trajectory
                            # TODO: not reasonable, we should use diffusion sampling for validation
                            logger(f"Using single-step prediction for validation...")
                            generated_trajectories_12d = val_model_out
                        else:
                            # Original diffusion sampling approach
                            logger(f"Using diffusion sampling for validation...")
                            generated_trajectories_12d = model.sample(
                                sequence_length=sequence_length,
                                batch_size=batch_size,
                                language_text=model_batch['object_category'],
                                bbox_corners=model_batch.get('bbox_corners', None),
                                device=device
                            )
                        
                        # Convert back to 9D for metrics calculation
                        generated_trajectories_9d = model.convert_12d_to_9d(generated_trajectories_12d)
                        
                        # Calculate metrics for each sample in batch
                        for i in range(batch_size):
                            if visualized_count >= vis_limit:
                                break
                                
                            # Extract ground truth data
                            gt_poses_9d = model_batch['poses'][i]  # Now 9D instead of 12D
                            gt_mask = model_batch['attention_mask'][i]
                            pred_poses_12d = generated_trajectories_12d[i]
                            
                            # Get actual length from mask
                            actual_length = torch.sum(gt_mask).int().item()
                            if actual_length < 2:
                                continue
                            
                            # Determine history length (following sparse conditioning)
                            if getattr(config, 'use_first_frame_only', False):
                                history_length = 1
                            else:
                                history_length = int(np.floor(actual_length * config.history_fraction))
                                history_length = max(1, min(history_length, actual_length - 1))
                            
                            # Extract position component for metrics (first 3 dimensions)
                            gt_positions = gt_poses_9d[:, :3]  # GT is 9D, extract xyz
                            pred_positions = pred_poses_12d[:, :3]  # Prediction is 12D, extract xyz
                            
                            # Split into history and future
                            gt_future_positions = gt_positions[history_length:actual_length]
                            pred_future_positions = pred_positions[history_length:actual_length]
                            future_mask = gt_mask[history_length:actual_length]
                            
                            if future_mask.sum() == 0:
                                continue
                            
                            # Compute metrics
                            l1_mean, rmse_ade, fde, frechet_distance, angular_cosine_similarity = compute_metrics_for_sample(
                                pred_future_positions, 
                                gt_future_positions, 
                                future_mask
                            )
                            
                            # Accumulate metrics
                            total_l1 += l1_mean.item()
                            total_rmse += rmse_ade.item()
                            total_fde += fde.item()
                            total_frechet += frechet_distance.item()
                            total_angular_cosine += angular_cosine_similarity.item()
                            total_valid_samples += 1
                            
                            # Visualization
                            if visualized_count < vis_limit:
                                # Extract data for visualization
                                vis_past_positions = gt_positions[:history_length]
                                vis_future_positions_gt = gt_positions[history_length:actual_length]
                                predicted_future_positions = pred_positions[history_length:actual_length]
                                
                                vis_past_mask = gt_mask[:history_length]
                                vis_future_mask_gt = gt_mask[history_length:actual_length]
                                
                                # Get metadata
                                obj_name = model_batch['object_name'][i] if 'object_name' in model_batch else f"unknown_{i}"
                                segment_idx = model_batch['segment_idx'][i].item() if 'segment_idx' in model_batch else None
                                
                                # Create filename and title
                                current_sequence_path = model_batch['sequence_path'][i] if 'sequence_path' in model_batch else f"seq_{i}"
                                sequence_name_extracted = os.path.splitext(os.path.basename(current_sequence_path))[0]
                                
                                filename_base = f"{obj_name}_seq_{sequence_name_extracted}_seg{segment_idx}_batch{batch_idx}"
                                vis_title_base = f"{obj_name} (Seq: {sequence_name_extracted}, Seg: {segment_idx}, Batch: {batch_idx})"
                                
                                # Only generate full trajectory visualization on first validation
                                if is_first_validation:
                                    full_traj_path = os.path.join(trajectory_vis_dir, f"{filename_base}_full_trajectory.png")
                                    gt_positions_vis = transform_coords_for_visualization(gt_positions.cpu())
                                    
                                    visualize_full_trajectory(
                                        positions=gt_positions_vis,
                                        attention_mask=gt_mask.cpu(),
                                        point_cloud=None,  # We don't have point cloud in this setup
                                        bbox_corners_sequence=None,
                                        trajectory_specific_bbox_info=None,
                                        trajectory_specific_bbox_mask=None,
                                        title=f"Full GT - {vis_title_base}",
                                        save_path=full_traj_path,
                                        segment_idx=segment_idx
                                    )
                                
                                # Always generate prediction vs ground truth visualization
                                pred_vs_gt_path = os.path.join(vis_output_dir, f"{filename_base}_prediction_vs_gt_epoch{epoch}.png")
                                
                                vis_past_positions_vis = transform_coords_for_visualization(vis_past_positions.cpu())
                                vis_future_positions_gt_vis = transform_coords_for_visualization(vis_future_positions_gt.cpu())
                                predicted_future_positions_vis = transform_coords_for_visualization(predicted_future_positions.cpu())
                                
                                visualize_prediction(
                                    past_positions=vis_past_positions_vis,
                                    future_positions_gt=vis_future_positions_gt_vis,
                                    future_positions_pred=predicted_future_positions_vis,
                                    past_mask=vis_past_mask.cpu(),
                                    future_mask_gt=vis_future_mask_gt.cpu(),
                                    title=f"Obj Traj Pred vs GT - {vis_title_base} (Epoch {epoch})",
                                    save_path=pred_vs_gt_path,
                                    segment_idx=segment_idx,
                                    show_orientation=False  # We can enable this later if needed
                                )
                                
                                visualized_count += 1
                    
                    except Exception as e:
                        logger(f"Warning: Error in trajectory sampling for validation: {e}")
                        
                # DEBUG: Only process first batch if config.first_batch is enabled
                if config.first_batch:
                    logger(f"DEBUG: Processed first validation batch (batch_idx={batch_idx}), breaking from validation loop")
                    break
                        
            except Exception as e:
                logger(f"Error processing validation batch {batch_idx}: {e}")
                # DEBUG: Still break after first batch even if there's an error and config.first_batch is enabled
                if config.first_batch:
                    logger(f"DEBUG: Breaking after first validation batch due to error")
                    break

    # Calculate average validation metrics
    if config.first_batch:
        num_processed_val_batches = 1  # DEBUG: We only process first batch when first_batch mode enabled
        logger(f"DEBUG: Validation loss calculated from {num_processed_val_batches} batch(es)")
    else:
        num_processed_val_batches = len(dataloader)
    avg_val_loss = val_total_loss / num_processed_val_batches
    val_metrics = {'total_loss': avg_val_loss}
    
    if total_valid_samples > 0:
        avg_l1 = total_l1 / total_valid_samples
        avg_rmse = total_rmse / total_valid_samples
        avg_fde = total_fde / total_valid_samples
        avg_frechet = total_frechet / total_valid_samples
        avg_angular_cosine = total_angular_cosine / total_valid_samples
        
        val_metrics.update({
            'l1_mean': avg_l1,
            'rmse': avg_rmse,
            'fde': avg_fde,
            'frechet': avg_frechet,
            'angular_cosine': avg_angular_cosine
        })
        
        print(f"Object Trajectory Validation Metrics - L1: {avg_l1:.4f}, RMSE: {avg_rmse:.4f}, FDE: {avg_fde:.4f}, Frechet: {avg_frechet:.4f}, Angular Cosine: {avg_angular_cosine:.4f}")
    
    return val_metrics


def main():
    
    # Configuration
    print("Loading CHOIS Object Trajectory Diffusion configuration...")
    config = CHOISObjectTrajectoryConfig().get_configs()
    
    # Check debug mode
    if config.first_batch:
        print("="*80)
        print("🚨 DEBUG MODE: ONLY PROCESSING FIRST BATCH OF EACH EPOCH 🚨")
        print("This is for debugging NaN issues. Use --first_batch to enable/disable.")
        print("="*80)
    
    # Add timestamp to save_path
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    original_save_path = config.save_path
    parent_dir = os.path.dirname(original_save_path)
    folder_name = os.path.basename(original_save_path)
    timestamped_folder_name = f"{timestamp}_obj_traj_{folder_name}"
    # config.save_path = os.path.join(parent_dir, timestamped_folder_name)
    config.save_path = original_save_path
    print(f"Original save path: {original_save_path}")

    print(f"Timestamped save path: {config.save_path}")
    
    print("Object Trajectory Diffusion Configuration loaded:")
    print(config)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Logging Setup
    os.makedirs(config.save_path, exist_ok=True)
    log_file = os.path.join(config.save_path, 'train_obj_traj_log.txt')
    def logger(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
    
    logger(f"Object Trajectory Diffusion Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger(f"Config: {vars(config)}\n")

    # WandB Initialization
    if config.wandb_mode != 'disabled':
        try:
            save_path_comment = os.path.basename(os.path.normpath(config.save_path))
            run_name = f"ObjTraj_{timestamp}_{save_path_comment}"
            wandb.init(
                project=config.wandb_project,
                config=vars(config),
                name=run_name,
                entity=config.wandb_entity,
                mode=config.wandb_mode
            )
            logger(f"WandB initialized successfully. Run name: {run_name}")
        except Exception as e:
            logger(f"Warning: Could not initialize WandB. Error: {e}")
            config.wandb_mode = 'disabled'
    else:
        logger("WandB is disabled.")

    # Dataset and DataLoader setup
    logger("Setting up dataset and dataloader...")

    train_sequences = []
    val_sequences = []
    loaded_from_files = False

    # Check for provided split files
    if config.train_split_file and config.val_split_file:
        if os.path.exists(config.train_split_file) and os.path.exists(config.val_split_file):
            logger(f"Loading train sequences from: {config.train_split_file}")
            logger(f"Loading validation sequences from: {config.val_split_file}")
            try:
                with open(config.train_split_file, 'r') as f:
                    train_sequences = [line.strip() for line in f if line.strip()]
                with open(config.val_split_file, 'r') as f:
                    val_sequences = [line.strip() for line in f if line.strip()]

                if train_sequences and val_sequences:
                    logger(f"Loaded {len(train_sequences)} train and {len(val_sequences)} validation sequences from files.")
                    loaded_from_files = True
                else:
                    logger("Warning: One or both provided split files are empty. Falling back to dynamic splitting.")
            except Exception as e:
                logger(f"Error loading split files: {e}. Falling back to dynamic splitting.")

    # Dynamic sequence splitting (fallback)
    if not loaded_from_files:
        logger("Proceeding with dynamic sequence splitting...")
        if not os.path.exists(config.adt_dataroot):
            logger(f"Error: adt_dataroot path {config.adt_dataroot} does not exist.")
            return

        if os.path.isdir(config.adt_dataroot) and HAS_SEQ_UTILS:
            logger(f"Scanning for sequences in {config.adt_dataroot}")
            try:
                all_sequences = find_adt_sequences(config.adt_dataroot)
                if not all_sequences:
                    logger(f"Error: No sequences found in {config.adt_dataroot}.")
                    return

                logger(f"Found {len(all_sequences)} total sequences. Splitting with train_ratio={config.train_ratio}")
                train_sequences, val_sequences = create_train_test_split(
                    all_sequences,
                    train_ratio=config.train_ratio,
                    random_seed=config.split_seed,
                    write_to_file=False
                )
                logger(f"Using {len(train_sequences)} sequences for training and {len(val_sequences)} for validation (dynamically split).")

                if config.train_ratio >= 1.0:
                    logger("Train ratio is >= 1.0, using all sequences for both training and validation.")
                    val_sequences = train_sequences

            except Exception as e:
                logger(f"Error during sequence finding/splitting: {e}. Please check adt_dataroot and utils.")
                return
        else:
            logger("Using single sequence or directory scan for training.")
            if os.path.isdir(config.adt_dataroot):
                all_items = [os.path.join(config.adt_dataroot, item) for item in os.listdir(config.adt_dataroot)]
                train_sequences = [item for item in all_items if os.path.isdir(item)]
            else:
                train_sequences = [config.adt_dataroot]
            val_sequences = train_sequences
            logger(f"Using {len(train_sequences)} sequences for training and validation.")

    # Save sequence lists
    if train_sequences and val_sequences:
        try:
            train_split_save_path = os.path.join(config.save_path, 'train_sequences.txt')
            val_split_save_path = os.path.join(config.save_path, 'val_sequences.txt')

            with open(train_split_save_path, 'w') as f:
                for seq_path in train_sequences:
                    f.write(f"{seq_path}\n")
            with open(val_split_save_path, 'w') as f:
                for seq_path in val_sequences:
                    f.write(f"{seq_path}\n")
            logger(f"Saved sequence lists to {train_split_save_path} and {val_split_save_path}")
        except Exception as e:
            logger(f"Warning: Could not save sequence lists: {e}")

    # Create datasets
    if getattr(config, 'global_cache_dir', None):
        cache_dir = config.global_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using global cache directory: {cache_dir}")
    else:
        cache_dir = os.path.join(config.save_path, 'trajectory_cache')
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using experiment-specific cache directory: {cache_dir}")
    
    train_dataset = GIMOMultiSequenceDataset(
        sequence_paths=train_sequences,
        config=config,
        cache_dir=cache_dir
    )
    val_dataset = GIMOMultiSequenceDataset(
        sequence_paths=val_sequences,
        config=config,
        cache_dir=cache_dir
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger("Error: Empty dataset. Check sequence paths and data.")
        return

    # Create DataLoaders
    collate_func = partial(gimo_collate_fn, dataset=train_dataset, num_sample_points=config.sample_points)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                             num_workers=config.num_workers, drop_last=True, collate_fn=collate_func)
    
    val_collate_func = partial(gimo_collate_fn, dataset=val_dataset, num_sample_points=config.sample_points)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                           num_workers=config.num_workers, drop_last=True, collate_fn=val_collate_func)
    
    logger(f"Train Dataset size: {len(train_dataset)}, Val Dataset size: {len(val_dataset)}")

    # Model initialization
    logger("Initializing Object Trajectory Diffusion Model...")
    
    # Initialize CLIP for text encoding
    clip_model = None
    if not getattr(config, 'no_text_embedding', False):
        try:
            clip_model, _ = clip.load("ViT-B/32", device=device)
            clip_model.eval()
            for param in clip_model.parameters():
                param.requires_grad = False
            logger("CLIP model loaded for text encoding")
        except Exception as e:
            logger(f"Warning: Could not load CLIP model: {e}")
    
    # Model parameters
    trajectory_length = config.trajectory_length
    
    model = ObjectTrajectoryDiffusion(
        d_feats=config.d_feats,  # 12D trajectory (xyz + 9D rotation matrix)
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
        use_bps=config.use_bps,
        bps_input_dim=config.bps_input_dim,
        bps_hidden_dim=config.bps_hidden_dim,
        bps_output_dim=config.bps_output_dim,
        bps_num_points=config.bps_num_points,
        use_text_embedding=config.text_embedding
    ).to(device)
    
    # Log model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger("\n=== OBJECT TRAJECTORY DIFFUSION MODEL ===")
    logger(f"Total Parameters: {total_params:,}")
    logger(f"Trainable Parameters: {trainable_params:,}")
    logger(f"Trajectory Length: {trajectory_length}")
    logger(f"Text Embedding Enabled: {config.text_embedding}")
    logger("=== END MODEL ARCHITECTURE ===\n")
    
    if config.wandb_mode != 'disabled':
        wandb.watch(model)

    # Optimizer and Scheduler
    logger("Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(config.adam_beta1, config.adam_beta2),
                           eps=config.adam_eps, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    logger(f"Optimizer: AdamW with lr={config.lr}")
    logger(f"Scheduler: ExponentialLR with gamma={config.gamma}")

    # Load checkpoint (if specified)
    start_epoch = 1
    best_val_loss = float('inf')
    if config.load_model_dir and os.path.exists(config.load_model_dir):
        logger(f"Loading model checkpoint from: {config.load_model_dir}")
        try:
            checkpoint = torch.load(config.load_model_dir, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            logger(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        except Exception as e:
            logger(f"Warning: Error loading checkpoint: {e}")

    # Training Loop
    debug_suffix = " (DEBUG: FIRST BATCH ONLY)" if config.first_batch else ""
    logger(f"\n--- Starting Object Trajectory Diffusion Training Loop{debug_suffix} ---")
    num_epochs = config.epoch
    
    for epoch in range(start_epoch, num_epochs + 1):
        logger(f"\nEpoch {epoch}/{num_epochs}")
        model.train()
        epoch_total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare batch for our model
                model_batch = prepare_batch_for_object_diffusion(batch, config, device)
                
                # Encode text features if available
                text_features = None
                if model_batch['object_category'] is not None and clip_model is not None:
                    text_features = encode_text_categories(
                        model_batch['object_category'], clip_model, device
                    )
                model_batch['text_features'] = text_features
                
                # Forward pass
                model_out, loss = model(model_batch, history_fraction=config.history_fraction, conditioning_strategy=config.conditioning_strategy, waypoint_interval=config.waypoint_interval)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_total_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # DEBUG: Only process first batch if config.first_batch is enabled
                if config.first_batch:
                    logger(f"DEBUG: Processed first batch (batch_idx={batch_idx}), breaking from training loop")
                    break
                
            except Exception as e:
                logger(f"Error processing batch {batch_idx}: {e}")
                # DEBUG: Still break after first batch even if there's an error and config.first_batch is enabled
                if config.first_batch:
                    logger(f"DEBUG: Breaking after first batch due to error")
                    break
        
        # Calculate average training loss
        if config.first_batch:
            num_processed_batches = 1  # DEBUG: We only process first batch when first_batch mode enabled
            logger(f"DEBUG: Training loss calculated from {num_processed_batches} batch(es)")
        else:
            num_processed_batches = len(train_loader)
        avg_train_loss = epoch_total_loss / num_processed_batches
        train_metrics = {'total_loss': avg_train_loss}
        
        log_metrics(epoch, "Training", train_metrics, logger)
        if config.wandb_mode != 'disabled':
            wandb.log({"train/" + k: v for k, v in train_metrics.items()}, step=epoch)
            wandb.log({"learning_rate": scheduler.get_last_lr()[0]}, step=epoch)
        
        # Validation
        val_freq = config.val_fre
        if epoch % val_freq == 0:
            val_metrics = validate_object_diffusion(model, val_loader, device, config, epoch, logger, clip_model, use_single_step_prediction=True)
            log_metrics(epoch, "Validation", val_metrics, logger)
            if config.wandb_mode != 'disabled':
                wandb.log({"val/" + k: v for k, v in val_metrics.items()}, step=epoch)
            
            # Save best model
            current_val_loss = val_metrics['total_loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                logger(f"---> New best validation loss: {best_val_loss:.4f}. Saving best model...")
                best_model_path = os.path.join(config.save_path, 'best_obj_traj_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': vars(config)
                }, best_model_path)
                if config.wandb_mode != 'disabled':
                    wandb.save(best_model_path)

        # Periodic checkpoint saving
        save_freq = config.save_fre
        if epoch % save_freq == 0:
            ckpt_path = os.path.join(config.save_path, f'ckpt_epoch_{epoch}.pth')
            logger(f"Saving periodic checkpoint to {ckpt_path}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': vars(config)
            }, ckpt_path)

        # LR Step
        scheduler.step()

    logger("\n--- Object Trajectory Diffusion Training Finished ---")
    
    # Save final model
    final_model_path = os.path.join(config.save_path, 'final_obj_traj_model.pth')
    logger(f"Saving final model to {final_model_path}")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config': vars(config)
    }, final_model_path)

    if config.wandb_mode != 'disabled':
        wandb.save(final_model_path)
        wandb.finish()

    logger(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if config.first_batch:
        print("\n" + "="*80)
        print("🚨 DEBUG MODE WAS ACTIVE - ONLY FIRST BATCH WAS PROCESSED 🚨")
        print("To run full training, use the script without --first_batch flag.")
        print("="*80)


if __name__ == "__main__":
    main() 