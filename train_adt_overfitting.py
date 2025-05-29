#!/usr/bin/env python3
import torch
import torch.optim as optim
import os
import time
import numpy as np
from tqdm import tqdm
import json
import wandb # Added WandB
from torch.utils.data.dataloader import default_collate # Import default_collate
from functools import partial # Import partial for binding args to collate_fn
# Add Dataset import
from torch.utils.data import Dataset, DataLoader
# Add Matplotlib imports for point cloud visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import traceback

# Imports from our project
from config.adt_config import ADTObjectMotionConfig
from dataset.gimo_multi_sequence_dataset import GIMOMultiSequenceDataset
from model.gimo_adt_model import GIMO_ADT_Model
# from torch.utils.data import DataLoader # Moved up
from utils.visualization import visualize_trajectory, visualize_prediction, visualize_full_trajectory # Import visualization utils

# Import metrics utilities
from utils.metrics_utils import (
    transform_coords_for_visualization,
    compute_metrics_for_sample,
    gimo_collate_fn
)

# --- Import ADT Sequence Utilities ---
try:
    from ariaworldgaussians.adt_sequence_utils import find_adt_sequences, create_train_test_split
    HAS_SEQ_UTILS = True
except ImportError:
    print("Warning: Could not import adt_sequence_utils. Sequence splitting requires adt_dataroot to point to pre-split directories or a single sequence.")
    HAS_SEQ_UTILS = False
# -------------------------------------

# Set random seed for reproducibility (optional but good practice)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def log_metrics(epoch, title, metrics, logger_func):
    log_str = f"Epoch {epoch} {title}: Total Loss {metrics['total_loss']:.4f}"
    log_str += " | Components: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'total_loss'])
    logger_func(log_str)

def validate(model, dataloader, device, config, epoch):
    model.eval() # Set model to evaluation mode
    val_total_loss = 0.0
    val_loss_components = {}
    visualized_count = 0 # Counter for visualizations this epoch
    vis_limit = config.num_val_visualizations
    
    # Check if this is the first validation run for special visualization handling
    is_first_validation = epoch == config.val_fre
    
    # Create different output directories based on whether it's the first validation
    vis_output_dir = os.path.join(config.save_path, "overfitting_val_visualizations", f"epoch_{epoch}")
    
    # Create special directory for trajectory and split visualizations (only used on first run)
    if is_first_validation:
        trajectory_vis_dir = os.path.join(config.save_path, "overfitting_trajectory_visualizations")
        os.makedirs(trajectory_vis_dir, exist_ok=True)
        print(f"First validation run: trajectory visualizations will be saved to: {trajectory_vis_dir}")
    
    if vis_limit > 0:
        os.makedirs(vis_output_dir, exist_ok=True)
        print(f"Validation visualizations will be saved to: {vis_output_dir}")

    # Add metrics tracking
    total_l1 = 0.0
    total_rmse = 0.0
    total_fde = 0.0
    total_valid_samples = 0

    print("\nRunning validation...")
    with torch.no_grad():
        # Since batch size is 1, no need for progress bar usually
        for batch_idx, batch in enumerate(dataloader):
            try:
                full_trajectory_batch = batch['full_poses'].float().to(device)
                point_cloud_batch = batch['point_cloud'].float().to(device) # Get point cloud from collated batch
                # Only get bbox_corners if not using no_bbox
                if not config.no_bbox:
                    bbox_corners_batch = batch['bbox_corners'].float().to(device) # Get bbox_corners
                else:
                    bbox_corners_batch = None
                
                # Move mask to device for loss calc
                batch['full_attention_mask'] = batch['full_attention_mask'].to(device)
                
                # --- Dynamically determine input history length based on actual trajectory length ---
                current_full_trajectory = full_trajectory_batch # Shape [1, config.trajectory_length, 6]
                current_attention_mask = batch['full_attention_mask'] # Shape [1, config.trajectory_length]
                
                actual_length = current_attention_mask[0].sum().int().item()

                if config.use_first_frame_only:
                    # Ensure dynamic_input_hist_len is at least 0 and at most actual_length
                    dynamic_input_hist_len = min(1, actual_length) if actual_length > 0 else 0
                    input_trajectory_batch = current_full_trajectory[:, :dynamic_input_hist_len, :]
                    if not config.no_bbox and bbox_corners_batch is not None:
                        bbox_corners_input_batch = bbox_corners_batch[:, :dynamic_input_hist_len, :, :]
                    else:
                        bbox_corners_input_batch = None
                else:
                    # Calculate history length based on actual_length and history_fraction
                    # Ensure it's at least 1 if actual_length > 0, and not more than actual_length
                    if actual_length > 0:
                        dynamic_input_hist_len = int(np.floor(actual_length * config.history_fraction))
                        dynamic_input_hist_len = max(1, dynamic_input_hist_len) # Ensure at least 1 if possible
                        dynamic_input_hist_len = min(dynamic_input_hist_len, actual_length) # Cap at actual_length
                    else:
                        dynamic_input_hist_len = 0 # No history if trajectory is empty
                    
                    input_trajectory_batch = current_full_trajectory[:, :dynamic_input_hist_len, :]
                    if not config.no_bbox and bbox_corners_batch is not None:
                        bbox_corners_input_batch = bbox_corners_batch[:, :dynamic_input_hist_len, :, :]
                    else:
                        bbox_corners_input_batch = None
                # --- End dynamic input history length determination ---

                # Get object category IDs if embedding is enabled
                if not config.no_text_embedding:
                    object_category_ids = batch['object_category_id'].to(device)
                else:
                    object_category_ids = None

            except KeyError as e: logger(f"Error: Missing key {e} in batch {batch_idx}. Skipping."); continue
            except Exception as e: logger(f"Error processing batch {batch_idx}: {e}. Skipping."); continue

            # Forward pass with input trajectory, point cloud, bbox corners, and category IDs
            predicted_full_trajectory = model(input_trajectory_batch, point_cloud_batch, bbox_corners_input_batch, object_category_ids)
            total_loss, loss_dict = model.compute_loss(predicted_full_trajectory, batch)

            val_total_loss += total_loss.item()
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    val_loss_components[key] = val_loss_components.get(key, 0.0) + value.item()

            # Optional: Log validation loss per step if desired for overfitting
            # print(f"Validation step loss: {total_loss.item():.4f}")

            # --- Calculate Additional Metrics for Each Sample in Batch ---
            batch_size = full_trajectory_batch.shape[0]
            for i in range(batch_size):
                # Extract ground truth and prediction
                gt_full_poses = batch['full_poses'][i]
                gt_full_mask = batch['full_attention_mask'][i]
                pred_full_trajectory = predicted_full_trajectory[i]
                
                # Get actual length from mask
                actual_length = torch.sum(gt_full_mask).int().item()
                if actual_length < 2:
                    continue  # Skip samples with too few valid points
                
                # Determine split between history and future
                if config.use_first_frame_only:
                    history_length = 1
                else:
                    history_length = int(np.floor(actual_length * config.history_fraction))
                    history_length = max(1, min(history_length, actual_length - 1))
                
                # Extract position components (first 3 dimensions)
                position_dim = config.object_position_dim
                gt_positions = gt_full_poses[:, :position_dim]
                pred_positions = pred_full_trajectory[:, :position_dim]
                
                # Split into history and future
                gt_future_positions = gt_positions[history_length:actual_length]
                pred_future_positions = pred_positions[history_length:actual_length]
                future_mask = gt_full_mask[history_length:actual_length]
                
                if future_mask.sum() == 0:
                    continue  # Skip if no valid future points
                
                # Compute metrics
                l1_mean, rmse_ade, fde = compute_metrics_for_sample(
                    pred_future_positions, 
                    gt_future_positions, 
                    future_mask
                )
                
                # Accumulate metrics
                total_l1 += l1_mean.item()
                total_rmse += rmse_ade.item()
                total_fde += fde.item()
                total_valid_samples += 1

            # --- Visualization Logic (Adapted for single sample) ---
            if visualized_count < vis_limit:
                # Batch size is always 1 here
                samples_to_vis = 1
                
                # Extract data for visualization (index 0)
                gt_full_poses_batch = batch['full_poses'] # Get full poses (positions + orientations)
                gt_full_mask_batch = batch.get('full_attention_mask')
                pred_full_trajectory_batch = predicted_full_trajectory # Predicted full trajectory
                
                for i in range(samples_to_vis): # Will only run once
                    if visualized_count >= vis_limit:
                        break
                        
                    # Get data for the current sample (the only sample)
                    gt_full_positions = gt_full_poses_batch[i, :, :3]
                    gt_full_rotations = gt_full_poses_batch[i, :, 3:]
                    gt_full_mask = gt_full_mask_batch[i] if gt_full_mask_batch is not None else None
                    pred_full_positions = pred_full_trajectory_batch[i, :, :3]
                    pred_full_rotations = pred_full_trajectory_batch[i, :, 3:]
                    sample_pointcloud = point_cloud_batch[i].cpu()  # Get point cloud for this sample
                    
                    # --- Dynamic Split Calculation for Visualization ---
                    if gt_full_mask is None:
                        print("Warning: Cannot perform dynamic split for visualization - mask missing.")
                        actual_length = gt_full_positions.shape[0]
                        if config.use_first_frame_only:
                            history_length_for_vis = 1
                        else:
                            history_length_for_vis = int(np.floor(actual_length * config.history_fraction))
                            history_length_for_vis = max(1, min(history_length_for_vis, actual_length))
                    else:
                        actual_length = torch.sum(gt_full_mask).int().item()
                        if config.use_first_frame_only:
                            history_length_for_vis = 1 if actual_length >= 1 else 0
                        else:
                            history_length_for_vis = int(np.floor(actual_length * config.history_fraction))
                            history_length_for_vis = max(1, min(history_length_for_vis, actual_length))

                    # Slice GT based on the calculated history length for visualization
                    vis_past_positions = gt_full_positions[:history_length_for_vis]
                    vis_future_positions_gt = gt_full_positions[history_length_for_vis:actual_length] # Slice up to actual length
                    
                    vis_past_rotations = gt_full_rotations[:history_length_for_vis]
                    vis_future_rotations_gt = gt_full_rotations[history_length_for_vis:actual_length]

                    vis_past_mask = gt_full_mask[:history_length_for_vis] if gt_full_mask is not None else None
                    vis_future_mask_gt = gt_full_mask[history_length_for_vis:actual_length] if gt_full_mask is not None else None

                    # Slice prediction dynamically based on use_first_frame_only
                    if config.use_first_frame_only:
                        valid_future_len = actual_length - history_length_for_vis
                        predicted_future_positions = pred_full_positions[:valid_future_len] 
                        predicted_future_rotations = pred_full_rotations[:valid_future_len]
                    else:
                        # Standard case: model output is full trajectory
                        pred_past_positions = pred_full_positions[:history_length_for_vis]
                        predicted_future_positions = pred_full_positions[history_length_for_vis:actual_length]
                        
                        pred_past_rotations = pred_full_rotations[:history_length_for_vis]
                        predicted_future_rotations = pred_full_rotations[history_length_for_vis:actual_length]
                    # ----------------------------------------------------

                    obj_name = batch['object_name'][i] if 'object_name' in batch else f"unknown_{i}"
                    segment_idx = batch['segment_idx'][i].item() if 'segment_idx' in batch and batch['segment_idx'][i].item() != -1 else None
                    
                    # --- MODIFICATION START for filename and title ---
                    # In overfitting, sequence info comes from first_sample, batch_idx is 0 for the dataloader
                    # `first_sample_seq_raw` should be defined earlier in the main() and accessible here or passed
                    # For simplicity, we assume sequence_name_extracted can be derived from `batch` if it carries it
                    # or from a variable holding the first_sample's sequence path.
                    # Let's assume `first_sample_seq_name_for_vis` is available (derived in main and passed or re-derived)
                    
                    # Re-deriving sequence name from batch (which contains the first_sample data)
                    current_sequence_path = batch['sequence_path'][i] # batch now contains sequence_path collated
                    sequence_name_extracted = os.path.splitext(os.path.basename(current_sequence_path))[0]
                    current_batch_idx = batch_idx # from enumerate(dataloader)

                    fn_obj_name = obj_name
                    fn_seq_name = f"seq_{sequence_name_extracted}"
                    fn_seg_id = f"seg{segment_idx}" if segment_idx is not None else "segNA"
                    fn_batch_id = f"batch{current_batch_idx}"

                    filename_base = f"{fn_obj_name}_{fn_seq_name}_{fn_seg_id}_{fn_batch_id}"

                    title_obj_name = obj_name
                    title_seq_name = f"Seq: {sequence_name_extracted}"
                    title_seg_id = f"Seg: {segment_idx if segment_idx is not None else 'NA'}"
                    title_batch_id = f"Batch: {current_batch_idx}"

                    vis_title_base = f"{title_obj_name} ({title_seq_name}, {title_seg_id}, {title_batch_id})"
                    # --- MODIFICATION END ---
                    
                    # Check if orientation visualization is enabled
                    show_ori_arrows = getattr(config, 'show_ori_arrows', False)
                    viz_ori_scale = getattr(config, 'viz_ori_scale', 0.2) # Added from train_adt.py

                    # Only generate full trajectory and split visualizations on first validation
                    if is_first_validation:
                        # Full Trajectory Visualization - now with point cloud and orientation
                        full_traj_path = os.path.join(trajectory_vis_dir, f"{filename_base}_full_trajectory_with_scene.png")
                        # Get bbox corners for current sample if available
                        sample_bbox_corners_cpu = None
                        if not config.no_bbox and bbox_corners_batch is not None: # Check if bbox_corners_batch is not None
                            sample_bbox_corners_cpu = bbox_corners_batch[i].cpu()  # Extract bbox corners for current sample
                        
                        # Apply transformations for visualization
                        gt_full_positions_vis = transform_coords_for_visualization(gt_full_positions.cpu())
                        sample_pointcloud_vis = transform_coords_for_visualization(sample_pointcloud.cpu())
                        sample_bbox_corners_vis = transform_coords_for_visualization(sample_bbox_corners_cpu)

                        visualize_full_trajectory(
                            positions=gt_full_positions_vis,
                            attention_mask=gt_full_mask.cpu() if gt_full_mask is not None else None,
                            point_cloud=sample_pointcloud_vis,  # Pass the point cloud
                            bbox_corners_sequence=sample_bbox_corners_vis,  # Add bbox corners data if available
                            title=f"Full GT - {vis_title_base}",
                            save_path=full_traj_path,
                            segment_idx=segment_idx
                        )
                        
                        # Split Trajectory Visualization (uses dynamically sliced data)
                        split_traj_path = os.path.join(trajectory_vis_dir, f"{filename_base}_trajectory_split.png")
                        vis_past_positions_vis = transform_coords_for_visualization(vis_past_positions.cpu())
                        vis_future_positions_gt_vis = transform_coords_for_visualization(vis_future_positions_gt.cpu())
                        visualize_trajectory(
                            past_positions=vis_past_positions_vis,
                            future_positions=vis_future_positions_gt_vis,
                            past_mask=vis_past_mask.cpu() if vis_past_mask is not None else None,
                            future_mask=vis_future_mask_gt.cpu() if vis_future_mask_gt is not None else None,
                            title=f"Split GT - {vis_title_base}",
                            save_path=split_traj_path,
                            segment_idx=segment_idx
                        )
                    
                    # Always generate the prediction vs ground truth visualization
                    pred_vs_gt_path = os.path.join(vis_output_dir, f"{filename_base}_prediction_vs_gt_epoch{epoch}.png")
                    
                    # Apply transformations for prediction visualization
                    vis_past_positions_pred_vis = transform_coords_for_visualization(vis_past_positions.cpu())
                    vis_future_positions_gt_pred_vis = transform_coords_for_visualization(vis_future_positions_gt.cpu())
                    predicted_future_positions_vis = transform_coords_for_visualization(predicted_future_positions.cpu())
                    
                    visualize_prediction(
                        past_positions=vis_past_positions_pred_vis,
                        future_positions_gt=vis_future_positions_gt_pred_vis,
                        future_positions_pred=predicted_future_positions_vis, # Use dynamically sliced prediction
                        past_mask=vis_past_mask.cpu() if vis_past_mask is not None else None,
                        future_mask_gt=vis_future_mask_gt.cpu() if vis_future_mask_gt is not None else None, # Corrected argument name
                        title=f"Pred vs GT - {vis_title_base} (Epoch {epoch})",
                        save_path=pred_vs_gt_path,
                        segment_idx=segment_idx,
                        show_orientation=show_ori_arrows,
                        past_orientations=vis_past_rotations.cpu(), # Rotations not transformed
                        future_orientations_gt=vis_future_rotations_gt.cpu(), # Rotations not transformed
                        future_orientations_pred=predicted_future_rotations.cpu(), # Rotations not transformed
                        # viz_ori_scale=viz_ori_scale # Added from train_adt.py, but need to check if visualize_prediction supports it
                    )
                    
                    visualized_count += 1
            # --- End Visualization Logic ---

    # Average loss calculation is simpler since len(dataloader) is 1
    avg_val_loss = val_total_loss
    avg_loss_components = val_loss_components # No division needed
    avg_loss_components['total_loss'] = avg_val_loss
    
    # Calculate average metrics if samples were processed
    if total_valid_samples > 0:
        avg_l1 = total_l1 / total_valid_samples
        avg_rmse = total_rmse / total_valid_samples
        avg_fde = total_fde / total_valid_samples
        
        # Add metrics to the returned components
        avg_loss_components['l1_mean'] = avg_l1
        avg_loss_components['rmse'] = avg_rmse
        avg_loss_components['fde'] = avg_fde
        
        print(f"Validation Metrics - L1: {avg_l1:.4f}, RMSE: {avg_rmse:.4f}, FDE: {avg_fde:.4f}")
    
    return avg_loss_components

# --- Single Sample Dataset Definition ---
class SingleSampleDataset(Dataset):
    """A dataset that holds and returns only a single data sample."""
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Always return the same sample, ignoring the index
        return self.sample
# -------------------------------------

def main():
    # --- Configuration ---
    print("Loading configuration...")
    config = ADTObjectMotionConfig().get_configs()
    print("Configuration loaded:")
    print(config)

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Logging Setup ---
    os.makedirs(config.save_path, exist_ok=True)
    # Adjust log file name for overfitting
    log_file = os.path.join(config.save_path, 'train_overfitting_log.txt')
    def logger(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
    logger(f"Overfitting Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger(f"Config: {vars(config)}\n")

    # --- WandB Initialization --- 
    if config.wandb_mode != 'disabled':
        try:
            # Extract the last part of save_path as the comment
            save_path_comment = os.path.basename(os.path.normpath(config.save_path))
            run_name = f"GIMO_ADT_{time.strftime('%Y%m%d_%H%M%S')}_overfit_{save_path_comment}"
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

    # --- Dataset and DataLoader --- 
    logger("Setting up dataset and dataloader...")

    # --- Original Sequence Loading Logic (START) ---
    train_sequences = []
    val_sequences = []
    loaded_from_files = False

    # --- Check for Provided Split Files ---
    if config.train_split_file and config.val_split_file:
        if os.path.exists(config.train_split_file) and os.path.exists(config.val_split_file):
            logger(f"Loading train sequences from: {config.train_split_file}")
            logger(f"Loading validation sequences from: {config.val_split_file}")
            try:
                with open(config.train_split_file, 'r') as f:
                    train_sequences = [line.strip() for line in f if line.strip()]
                with open(config.val_split_file, 'r') as f:
                    val_sequences = [line.strip() for line in f if line.strip()]

                if not train_sequences or not val_sequences:
                    logger("Warning: One or both provided split files are empty. Falling back to dynamic splitting.")
                else:
                    logger(f"Loaded {len(train_sequences)} train and {len(val_sequences)} validation sequences from files.")
                    loaded_from_files = True
            except Exception as e:
                logger(f"Error loading split files: {e}. Falling back to dynamic splitting.")
        else:
            logger("Warning: Train/Validation split files provided but not found. Falling back to dynamic splitting.")

    # --- Dynamic Sequence Splitting (Fallback) ---
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
        elif os.path.isdir(config.adt_dataroot) and not HAS_SEQ_UTILS:
             logger("Warning: adt_sequence_utils not found. Assuming adt_dataroot contains only training sequences.")
             all_items = [os.path.join(config.adt_dataroot, item) for item in os.listdir(config.adt_dataroot)]
             train_sequences = [item for item in all_items if os.path.isdir(item)]
             val_sequences = train_sequences
             if not train_sequences:
                 logger(f"Error: No sequence directories found in {config.adt_dataroot} for training."); return
             logger(f"Using {len(train_sequences)} sequences for training and validation (from directory scan).")
        elif os.path.exists(config.adt_dataroot):
             logger(f"Using single sequence {config.adt_dataroot} for training and validation.")
             train_sequences = [config.adt_dataroot]
             val_sequences = [config.adt_dataroot]
        else:
             logger(f"Error: Invalid adt_dataroot path {config.adt_dataroot}.")
             return

    # --- Save the final train and validation sequence lists ---
    if train_sequences and val_sequences:
        try:
            os.makedirs(config.save_path, exist_ok=True)
            train_split_save_path = os.path.join(config.save_path, 'overfitting_train_sequences.txt') # Adjusted filename
            val_split_save_path = os.path.join(config.save_path, 'overfitting_val_sequences.txt') # Adjusted filename

            with open(train_split_save_path, 'w') as f:
                for seq_path in train_sequences:
                    f.write(f"{seq_path}\n")
            logger(f"Saved training sequence list ({len(train_sequences)} sequences) to {train_split_save_path}")

            with open(val_split_save_path, 'w') as f:
                for seq_path in val_sequences:
                    f.write(f"{seq_path}\n")
            logger(f"Saved validation sequence list ({len(val_sequences)} sequences) to {val_split_save_path}")

        except Exception as e:
             logger(f"Warning: Could not save final sequence lists: {e}")
    else:
        logger("Error: No train or validation sequences were determined. Cannot proceed.")
        return
    # --- Original Sequence Loading Logic (END) ---

    # Create original datasets (needed for collate_fn and extracting the first sample)
    cache_dir = os.path.join(config.save_path, 'trajectory_cache')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    
    original_train_dataset = GIMOMultiSequenceDataset(
        sequence_paths=train_sequences,
        config=config,
        cache_dir=cache_dir
    )
    original_val_dataset = GIMOMultiSequenceDataset(
        sequence_paths=val_sequences,
        config=config,
        cache_dir=cache_dir
    )

    if len(original_train_dataset) == 0:
        logger("Error: Original Training dataset is empty. Check sequence paths and data."); return
    # Validation dataset might be empty if train_ratio=1, which is fine for overfitting
    # if len(original_val_dataset) == 0:
    #      logger("Error: Original Validation dataset is empty. Check sequence paths and data."); return

    logger(f"Original Train Dataset size: {len(original_train_dataset)}")
    logger(f"Original Val Dataset size: {len(original_val_dataset)}")

    # --- Extract the first sample --- 
    logger("Extracting the first trajectory sample for overfitting...")
    first_sample = None
    target_object_name_part = "coffeecanistersmall" # Lowercase for case-insensitive search

    if len(original_train_dataset) > 0:
        for i in range(len(original_train_dataset)):
            sample = original_train_dataset[i]
            object_name = sample.get('object_name', "")
            if target_object_name_part in object_name.lower():
                first_sample = sample
                logger(f"Found sample containing '{target_object_name_part}' for overfitting: Object '{object_name}' at index {i}.")
                break # Stop after finding the first match

        if first_sample is None:
            logger(f"Warning: No sample with '{target_object_name_part}' in object_name found. Using the first available sample for overfitting.")
            first_sample = original_train_dataset[0]
    else:
        logger("Error: Original training dataset is empty. Cannot select a sample for overfitting.")
        return

    if first_sample is None: # Should only be reached if dataset was empty and logic above failed
        logger("Error: Could not select any sample for overfitting. Critical issue in sample selection.")
        return

    # Log details about the selected sample
    first_sample_name = first_sample.get('object_name', 'Unknown Object')
    first_sample_seq_raw = first_sample.get('sequence', 'Unknown_Sequence') # Get raw sequence path/name
    # pointcloud number of points
    first_sample_pointcloud_num_points = first_sample.get('pointcloud_num_points', -1)
    first_sample_seq_name = os.path.basename(first_sample_seq_raw) # Extract base name
    first_sample_ds_idx = first_sample.get('dataset_idx', -1)
    logger(f"Using sample: Object '{first_sample_name}' from Sequence '{first_sample_seq_name}' (Dataset Index: {first_sample_ds_idx})")
    # --------------------------------------------------------------------------

    # --- Create Single-Sample Datasets --- 
    train_dataset = SingleSampleDataset(first_sample)
    val_dataset = SingleSampleDataset(first_sample) # Use the same sample for validation

    # --- Create DataLoaders for Overfitting --- 
    # Bind the *original* dataset instance to the collate function
    collate_func = partial(gimo_collate_fn, dataset=original_train_dataset, num_sample_points=config.sample_points)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_func)
    
    # Use the *original* validation dataset for the validation collate function
    val_collate_func = partial(gimo_collate_fn, dataset=original_val_dataset, num_sample_points=config.sample_points)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, collate_fn=val_collate_func)
    
    # Update logging for dataset size
    logger(f"Overfitting Train Dataset size: {len(train_dataset)}, Val Dataset size: {len(val_dataset)}. DataLoaders ready.")

    # --- Model Initialization --- 
    logger("Initializing model...")
    if config.timestep == 0:
        logger("Using GIMO_ADT_Model (non-autoregressive).")
        model = GIMO_ADT_Model(config).to(device)
    elif config.timestep == 1:
        logger("Using GIMO_ADT_Autoregressive_Model.")
        from model.gimo_adt_autoregressive_model import GIMO_ADT_Autoregressive_Model # Import here
        model = GIMO_ADT_Autoregressive_Model(config).to(device)
    else:
        logger(f"Error: Invalid timestep configuration: {config.timestep}. Must be 0 or 1.")
        raise ValueError(f"Invalid timestep configuration: {config.timestep}")
    
    # --- Log Model Architecture ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger("\n=== MODEL ARCHITECTURE ===")
    logger(f"Total Parameters: {total_params:,}")
    logger(f"Trainable Parameters: {trainable_params:,}")
    logger(f"Text Embedding Enabled: {not getattr(config, 'no_text_embedding', False)}")
    logger(f"Bounding Box Processing Enabled: {not getattr(config, 'no_bbox', False)}")
    logger(f"Overfitting Mode: True")
    
    # # Log model components
    # components = {
    #     'Motion Linear': (model.motion_linear, []), 
    #     'Scene Encoder': (model.scene_encoder, ['hparams']), 
    # }
    
    # # Add bounding box related components if enabled
    # if not getattr(config, 'no_bbox', False):
    #     components.update({
    #         'FP Layer': (model.fp_layer, []),
    #         'BBox PointNet': (model.bbox_pointnet, ['conv1']), 
    #     })
    
    # components.update({
    #     'Motion BBox Encoder': (model.motion_bbox_encoder, ['n_input_channels', 'n_latent_channels', 'n_self_att_heads', 'n_self_att_layers']),
    #     'Embedding Layer (Fusion 2)': (model.embedding_layer, ['in_features', 'out_features']),
    #     'Output Encoder': (model.output_encoder, ['n_input_channels', 'n_latent_channels', 'n_self_att_heads', 'n_self_att_layers']),
    #     'Output Layer': (model.outputlayer, ['in_features', 'out_features'])
    # })
    
    # # Add text embedding components if enabled
    # # Check if text embedding is enabled and the attribute exists
    # if not getattr(config, 'no_text_embedding', False) and hasattr(model, 'category_embedding'):
    #     components.update({
    #         'Category Embedding': (model.category_embedding, ['num_embeddings', 'embedding_dim'])
    #     })
    
    # # Log each component's structure and parameters
    # for name, (component, attrs) in components.items():
    #     if component is None: # Add a check for None components
    #         logger(f"\n{name}: Not used/defined in this configuration.")
    #         continue
    #     params = sum(p.numel() for p in component.parameters())
    #     logger(f"\n{name}:")
    #     logger(f"  Parameters: {params:,}")
    #     try:
    #         # Try to log attributes if available
    #         for attr in attrs:
    #             if hasattr(component, attr):
    #                 logger(f"  {attr}: {getattr(component, attr)}")
    #     except:
    #         pass
    
    logger(f"\nOverfitting on sample:")
    logger(f"  Object: {first_sample_name}")
    logger(f"  Sequence: {first_sample_seq_name}")
    logger(f"  Category: {first_sample.get('object_category', 'Unknown')}")
    
    logger("=== END MODEL ARCHITECTURE ===\n")
    
    if config.wandb_mode != 'disabled':
        wandb.watch(model)

    # --- Optimizer and Scheduler --- 
    logger("Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    logger(f"Optimizer: AdamW with lr={config.lr}, beta1={config.adam_beta1}, beta2={config.adam_beta2}, eps={config.adam_eps}, weight_decay={config.weight_decay}")
    logger(f"Scheduler: ExponentialLR with gamma={config.gamma}")

    # --- Create directory for mask visualizations ---
    mask_check_dir = os.path.join(config.save_path, "mask_check")
    os.makedirs(mask_check_dir, exist_ok=True)
    logger(f"Mask visualizations will be saved to: {mask_check_dir}")

    # --- Load Checkpoint (If specified) --- 
    start_epoch = 1
    best_val_loss = float('inf')
    if config.load_model_dir:
        if os.path.exists(config.load_model_dir):
            logger(f"Loading model checkpoint from: {config.load_model_dir}")
            checkpoint = torch.load(config.load_model_dir, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and config.load_optim_dir is None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger("Loaded optimizer state from model checkpoint.")
            if 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 logger("Loaded scheduler state from model checkpoint.")
            if 'epoch' in checkpoint:
                 start_epoch = checkpoint['epoch'] + 1
                 logger(f"Resuming training from epoch {start_epoch}")
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                logger(f"Loaded best validation loss: {best_val_loss:.4f}")
        else:
            logger(f"Warning: load_model_dir specified but path not found: {config.load_model_dir}")
    if config.load_optim_dir and os.path.exists(config.load_optim_dir):
         logger(f"Loading optimizer state from: {config.load_optim_dir}")
         optimizer.load_state_dict(torch.load(config.load_optim_dir, map_location=device))

    # --- Training Loop --- 
    logger("\n--- Starting Overfitting Training Loop ---")
    num_epochs = config.epoch
    
    for epoch in range(start_epoch, num_epochs + 1):
        logger(f"\nEpoch {epoch}/{num_epochs}")
        model.train()
        epoch_total_loss = 0.0
        epoch_loss_components = {}

        # No progress bar needed for single sample training
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")

        # Loop runs only once
        for batch_idx, batch in enumerate(train_loader):
            try:
                full_trajectory_batch = batch['full_poses'].float().to(device) # Shape [1, config.trajectory_length, 6]
                print(f"full_trajectory_batch shape: {full_trajectory_batch.shape}")
                point_cloud_batch = batch['point_cloud'].float().to(device) # Shape [1, config.num_sample_points, 3]
                print(f"point_cloud_batch shape: {point_cloud_batch.shape}")
                # Only get bbox_corners if not using no_bbox
                if not config.no_bbox:
                    bbox_corners_batch = batch['bbox_corners'].float().to(device) # Get bbox_corners
                    if bbox_corners_batch is not None: print(f"bbox_corners_batch shape: {bbox_corners_batch.shape}")
                else:
                    bbox_corners_batch = None
                
                batch['full_attention_mask'] = batch['full_attention_mask'].to(device) # Shape [1, config.trajectory_length]
                print(f"batch['full_attention_mask'] shape: {batch['full_attention_mask'].shape}")

                # --- Dynamically determine input history length based on actual trajectory length ---
                current_full_trajectory = full_trajectory_batch # Shape [1, config.trajectory_length, 6]
                print(f"current_full_trajectory shape: {current_full_trajectory.shape}")
                current_attention_mask = batch['full_attention_mask'] # Shape [1, config.trajectory_length]
                print(f"current_attention_mask shape: {current_attention_mask.shape}")

                actual_length = current_attention_mask[0].sum().int().item() # Shape [1]
                print(f"actual_length: {actual_length}")

                if config.use_first_frame_only:
                    # Ensure dynamic_input_hist_len is at least 0 and at most actual_length
                    dynamic_input_hist_len = min(1, actual_length) if actual_length > 0 else 0
                    input_trajectory_batch = current_full_trajectory[:, :dynamic_input_hist_len, :]
                    if not config.no_bbox and bbox_corners_batch is not None:
                        bbox_corners_input_batch = bbox_corners_batch[:, :dynamic_input_hist_len, :, :]
                    else:
                        bbox_corners_input_batch = None
                else:
                    # Calculate history length based on actual_length and history_fraction
                    # Ensure it's at least 1 if actual_length > 0, and not more than actual_length
                    if actual_length > 0:
                        dynamic_input_hist_len = int(np.floor(actual_length * config.history_fraction)) # 91 * 0.3 = 27
                        dynamic_input_hist_len = max(1, dynamic_input_hist_len) # Ensure at least 1 if possible
                        dynamic_input_hist_len = min(dynamic_input_hist_len, actual_length) # Cap at actual_length
                    else:
                        dynamic_input_hist_len = 0 # No history if trajectory is empty

                    input_trajectory_batch = current_full_trajectory[:, :dynamic_input_hist_len, :] 
                    if not config.no_bbox and bbox_corners_batch is not None:
                        bbox_corners_input_batch = bbox_corners_batch[:, :dynamic_input_hist_len, :, :]
                    else:
                        bbox_corners_input_batch = None
                
                print(f"dynamic_input_hist_len: {dynamic_input_hist_len}")
                print(f"input_trajectory_batch shape: {input_trajectory_batch.shape}")
                if bbox_corners_input_batch is not None: print(f"bbox_corners_input_batch shape: {bbox_corners_input_batch.shape}")
                 # --- End dynamic input history length determination ---

                # Get object category IDs if embedding is enabled
                if not config.no_text_embedding:
                    object_category_ids = batch['object_category_id'].to(device)
                    if object_category_ids is not None: print(f"object_category_ids shape: {object_category_ids.shape}")
                else:
                    object_category_ids = None

            except KeyError as e: logger(f"Error: Missing key {e} in batch {batch_idx}. Skipping."); continue
            except Exception as e: logger(f"Error processing batch {batch_idx}: {e}. Skipping."); continue

            # Forward pass with input trajectory, point cloud, bbox corners, and category IDs
            predicted_full_trajectory = model(input_trajectory_batch, point_cloud_batch, bbox_corners_input_batch, object_category_ids)
            if predicted_full_trajectory is not None: print(f"predicted_full_trajectory shape: {predicted_full_trajectory.shape}")

            # --- Compute Loss with Visualization for the first epoch ---
            if epoch == start_epoch:
                sample_name_for_vis = f"{first_sample_name.replace(' ', '_')}_{first_sample_seq_name.replace('.json', '')}"
                total_loss, loss_dict = model.compute_loss(
                    predicted_full_trajectory, 
                    batch, 
                    epoch=epoch, 
                    batch_idx=batch_idx, # This will be 0 for overfitting train_loader
                    vis_save_dir=mask_check_dir, 
                    sample_name_for_vis=sample_name_for_vis
                )
            else:
                total_loss, loss_dict = model.compute_loss(predicted_full_trajectory, batch)
            # -----------------------------------------------------------

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    epoch_loss_components[key] = epoch_loss_components.get(key, 0.0) + value.item()

            # Log loss immediately
            # logger(f"Epoch {epoch} Training Step Loss: {total_loss.item():.4f}")

        # Average loss calculation is simpler (len(train_loader) is 1)
        avg_train_loss = epoch_total_loss
        avg_train_components = epoch_loss_components # No division needed
        avg_train_components['total_loss'] = avg_train_loss
        log_metrics(epoch, "Training", avg_train_components, logger)
        if config.wandb_mode != 'disabled':
             wandb.log({"train/" + k: v for k, v in avg_train_components.items()}, step=epoch)
             wandb.log({"learning_rate": scheduler.get_last_lr()[0]}, step=epoch)

        # --- Validation --- 
        if epoch % config.val_fre == 0:
            val_metrics = validate(model, val_loader, device, config, epoch)
            log_metrics(epoch, "Validation", val_metrics, logger)
            if config.wandb_mode != 'disabled':
                wandb.log({"val/" + k: v for k, v in val_metrics.items()}, step=epoch)
                if config.num_val_visualizations > 0:
                    # For the first validation run, log both trajectory visualizations and prediction visualizations
                    if epoch == config.val_fre:
                        trajectory_vis_dir = os.path.join(config.save_path, "overfitting_trajectory_visualizations")
                        if os.path.exists(trajectory_vis_dir):
                            try:
                                # Log trajectory visualizations (full trajectory and split)
                                traj_images = [img for img in os.listdir(trajectory_vis_dir) 
                                             if img.endswith('.png')]
                                
                                if traj_images: # Simplified: log all trajectory images for the single sample
                                    wandb.log({"trajectory_visualizations": [wandb.Image(os.path.join(trajectory_vis_dir, img)) 
                                                                           for img in sorted(traj_images)]}, step=epoch)
                            except Exception as e:
                                logger(f"Warning: Failed to log trajectory visualizations to WandB: {e}")
                    
                    # Always log prediction vs ground truth visualizations
                    vis_output_dir = os.path.join(config.save_path, "overfitting_val_visualizations", f"epoch_{epoch}")
                    if os.path.exists(vis_output_dir):
                        try:
                            # Only log prediction vs GT visualizations 
                            pred_images = [img for img in os.listdir(vis_output_dir) 
                                          if img.endswith('.png') and 'prediction_vs_gt' in img]
                            
                            if pred_images: # Simplified: log all prediction images for the single sample
                                wandb.log({"prediction_visualizations": [wandb.Image(os.path.join(vis_output_dir, img)) 
                                                                       for img in sorted(pred_images)]}, step=epoch)
                        except Exception as e:
                            logger(f"Warning: Failed to log prediction visualizations to WandB: {e}")

            current_val_loss = val_metrics['total_loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                logger(f"---> New best validation loss: {best_val_loss:.4f}. Saving best model...")
                best_model_path = os.path.join(config.save_path, 'best_overfitting_model.pth') # Adjusted filename
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': vars(config),
                    # Adjust sample details to save base sequence name
                    'overfitting_sample_details': {'name': first_sample_name, 'sequence': first_sample_seq_name, 'dataset_idx': first_sample_ds_idx} # Save details of the sample
                }, best_model_path)
                if config.wandb_mode != 'disabled':
                     wandb.save(best_model_path)

        # --- Periodic Checkpoint Saving --- 
        if epoch % config.save_fre == 0:
            ckpt_path = os.path.join(config.save_path, f'ckpt_overfitting_epoch_{epoch}.pth') # Adjusted filename
            logger(f"Saving periodic checkpoint to {ckpt_path}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': vars(config),
                # Adjust sample details to save base sequence name
                'overfitting_sample_details': {'name': first_sample_name, 'sequence': first_sample_seq_name, 'dataset_idx': first_sample_ds_idx}
            }, ckpt_path)

        # --- LR Step --- 
        scheduler.step()

    logger("\n--- Overfitting Training Finished ---")
    final_model_path = os.path.join(config.save_path, 'final_overfitting_model.pth') # Adjusted filename
    logger(f"Saving final model to {final_model_path}")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config': vars(config),
        # Adjust sample details to save base sequence name
        'overfitting_sample_details': {'name': first_sample_name, 'sequence': first_sample_seq_name, 'dataset_idx': first_sample_ds_idx}
    }, final_model_path)

    if config.wandb_mode != 'disabled':
        wandb.save(final_model_path)
        wandb.finish()

if __name__ == '__main__':
    main() 