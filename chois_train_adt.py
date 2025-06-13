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
# Add Matplotlib imports for point cloud visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Imports from our project
from config.adt_config import ADTObjectMotionConfig
from dataset.gimo_multi_sequence_dataset import GIMOMultiSequenceDataset
from model.chois_gimo_adt_model import SimpleTrajectoryTransformer
from torch.utils.data import DataLoader
from utils.visualization import visualize_trajectory, visualize_prediction, visualize_full_trajectory # Import visualization utils

# Import metrics utilities
from utils.metrics_utils import (
    transform_coords_for_visualization,
    compute_metrics_for_sample,
    gimo_collate_fn,
    plot_orientation_distribution,
    plot_gradient_magnitudes,
    GradientTracker
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
    vis_output_dir = os.path.join(config.save_path, "val_visualizations", f"epoch_{epoch}")
    
    # Create special directory for trajectory and split visualizations (only used on first run)
    if is_first_validation:
        trajectory_vis_dir = os.path.join(config.save_path, "trajectory_visualizations")
        os.makedirs(trajectory_vis_dir, exist_ok=True)
        print(f"First validation run: trajectory visualizations will be saved to: {trajectory_vis_dir}")
    
    if vis_limit > 0:
        os.makedirs(vis_output_dir, exist_ok=True)
        print(f"Validation visualizations will be saved to: {vis_output_dir}")

    # Add metrics tracking
    total_l1 = 0.0
    total_rmse = 0.0
    total_fde = 0.0
    total_frechet = 0.0
    total_angular_cosine = 0.0
    total_valid_samples = 0

    print("\nRunning validation...")
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Compatibility: rename 'full_poses' to 'poses' if it exists
                if 'full_poses' in batch and 'poses' not in batch:
                    batch['poses'] = batch.pop('full_poses')
                if 'full_attention_mask' in batch and 'attention_mask' not in batch:
                    batch['attention_mask'] = batch.pop('full_attention_mask')

                # Move necessary tensors to device
                batch['poses'] = batch['poses'].float().to(device)
                batch['attention_mask'] = batch['attention_mask'].to(device)
                # object_category is a list of strings and stays on CPU
            except KeyError as e: logger(f"Error: Missing key {e} in batch {batch_idx}. Skipping."); continue
            except Exception as e: logger(f"Error processing batch {batch_idx}: {e}. Skipping."); continue

            # 1. Forward pass - model now takes the whole batch dictionary
            predicted_full_trajectory, cond_mask = model(batch, history_fraction=config.history_fraction, conditioning_strategy=config.conditioning_strategy, waypoint_interval=config.waypoint_interval)

            # 2. Compute loss
            total_loss, loss_dict = model.compute_loss(predicted_full_trajectory, cond_mask, batch)

            val_total_loss += total_loss.item()
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    val_loss_components[key] = val_loss_components.get(key, 0.0) + value.item()

            progress_bar.set_postfix({'val_loss': f"{total_loss.item():.4f}"})

            # --- Calculate Additional Metrics for Each Sample in Batch ---
            batch_size = batch['poses'].shape[0]
            for i in range(batch_size):
                # Extract ground truth and prediction
                gt_full_poses = batch['poses'][i]
                gt_full_mask = batch['attention_mask'][i]
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
                
                # Extract position component (first 3 dimensions) for metrics
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

            # --- Visualization Logic ---
            if visualized_count < vis_limit:
                batch_size = batch['poses'].shape[0]
                samples_to_vis = min(batch_size, vis_limit - visualized_count)
                
                # Extract data for visualization
                gt_full_poses_batch = batch['poses'] # Get full poses
                gt_full_mask_batch = batch.get('attention_mask')
                pred_full_trajectory_batch = predicted_full_trajectory # Renamed for clarity
                point_cloud_batch = batch['point_cloud'] # For visualization context
                
                for i in range(samples_to_vis):
                    if visualized_count >= vis_limit:
                        break # Ensure we don't exceed limit within the inner loop
                        
                    # Get data for the current sample
                    gt_full_poses = gt_full_poses_batch[i]
                    gt_full_mask = gt_full_mask_batch[i] if gt_full_mask_batch is not None else None
                    pred_full_trajectory = pred_full_trajectory_batch[i]
                    sample_pointcloud = point_cloud_batch[i].cpu()  # Get point cloud for this sample
                    
                    # --- Dynamic Split Calculation for Visualization ---
                    if gt_full_mask is None:
                        print("Warning: Cannot perform dynamic split for visualization - mask missing.")
                        actual_length = gt_full_poses.shape[0]
                        history_length_for_vis = int(np.floor(actual_length * config.history_fraction))
                        history_length_for_vis = max(1, min(history_length_for_vis, actual_length))
                    else:
                        actual_length = torch.sum(gt_full_mask).int().item()
                        history_length_for_vis = int(np.floor(actual_length * config.history_fraction))
                        history_length_for_vis = max(1, min(history_length_for_vis, actual_length)) if actual_length > 0 else 0

                    # Extract position and orientation components
                    position_dim = 3
                    
                    gt_full_positions = gt_full_poses[:, :position_dim]
                    gt_full_rotations = gt_full_poses[:, position_dim:]
                    
                    pred_full_positions = pred_full_trajectory[:, :position_dim]
                    pred_full_rotations = pred_full_trajectory[:, position_dim:]

                    # Slice GT based on the calculated history length for visualization
                    vis_past_positions = gt_full_positions[:history_length_for_vis]
                    vis_future_positions_gt = gt_full_positions[history_length_for_vis:actual_length]
                    
                    vis_past_rotations = gt_full_rotations[:history_length_for_vis]
                    vis_future_rotations_gt = gt_full_rotations[history_length_for_vis:actual_length]

                    vis_past_mask = gt_full_mask[:history_length_for_vis] if gt_full_mask is not None else None
                    vis_future_mask_gt = gt_full_mask[history_length_for_vis:actual_length] if gt_full_mask is not None else None

                    predicted_future_positions = pred_full_positions[history_length_for_vis:actual_length]
                    predicted_future_rotations = pred_full_rotations[history_length_for_vis:actual_length]
                    # ----------------------------------------------------

                    obj_name = batch['object_name'][i] if 'object_name' in batch else f"unknown_{i}"
                    segment_idx = batch['segment_idx'][i].item() if 'segment_idx' in batch and batch['segment_idx'][i].item() != -1 else None
                    
                    current_sequence_path = batch['sequence_path'][i]
                    sequence_name_extracted = os.path.splitext(os.path.basename(current_sequence_path))[0]
                    current_batch_idx = batch_idx

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
                    
                    show_ori_arrows = getattr(config, 'show_ori_arrows', False)
                    
                    if is_first_validation:
                        full_traj_path = os.path.join(trajectory_vis_dir, f"{filename_base}_full_trajectory_with_scene.png")
                        
                        gt_full_positions_vis = transform_coords_for_visualization(gt_full_positions.cpu())
                        sample_pointcloud_vis = transform_coords_for_visualization(sample_pointcloud.cpu())

                        visualize_full_trajectory(
                            positions=gt_full_positions_vis,
                            attention_mask=gt_full_mask.cpu() if gt_full_mask is not None else None,
                            point_cloud=sample_pointcloud_vis,
                            title=f"Full GT - {vis_title_base}",
                            save_path=full_traj_path,
                            segment_idx=segment_idx
                        )
                        
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
                    
                    pred_vs_gt_path = os.path.join(vis_output_dir, f"{filename_base}_prediction_vs_gt_epoch{epoch}.png")
                    
                    vis_past_positions_pred_vis = transform_coords_for_visualization(vis_past_positions.cpu())
                    vis_future_positions_gt_pred_vis = transform_coords_for_visualization(vis_future_positions_gt.cpu())
                    predicted_future_positions_vis = transform_coords_for_visualization(predicted_future_positions.cpu())
                    
                    visualize_prediction(
                        past_positions=vis_past_positions_pred_vis,
                        future_positions_gt=vis_future_positions_gt_pred_vis,
                        future_positions_pred=predicted_future_positions_vis,
                        past_mask=vis_past_mask.cpu() if vis_past_mask is not None else None,
                        future_mask_gt=vis_future_mask_gt.cpu() if vis_future_mask_gt is not None else None,
                        title=f"Pred vs GT - {vis_title_base} (Epoch {epoch})",
                        save_path=pred_vs_gt_path,
                        segment_idx=segment_idx,
                        show_orientation=show_ori_arrows,
                        past_orientations=vis_past_rotations.cpu(),
                        future_orientations_gt=vis_future_rotations_gt.cpu(),
                        future_orientations_pred=predicted_future_rotations.cpu()
                    )
                    
                    visualized_count += 1
            # --- End Visualization Logic ---

    avg_val_loss = val_total_loss / len(dataloader)
    avg_loss_components = {k: v / len(dataloader) for k, v in val_loss_components.items()}
    avg_loss_components['total_loss'] = avg_val_loss
    
    if total_valid_samples > 0:
        avg_l1 = total_l1 / total_valid_samples
        avg_rmse = total_rmse / total_valid_samples
        avg_fde = total_fde / total_valid_samples
        avg_frechet = total_frechet / total_valid_samples
        avg_angular_cosine = total_angular_cosine / total_valid_samples
        
        avg_loss_components['l1_mean'] = avg_l1
        avg_loss_components['rmse'] = avg_rmse
        avg_loss_components['fde'] = avg_fde
        avg_loss_components['frechet'] = avg_frechet
        avg_loss_components['angular_cosine'] = avg_angular_cosine
        
        print(f"Validation Metrics - L1: {avg_l1:.4f}, RMSE: {avg_rmse:.4f}, FDE: {avg_fde:.4f}, Frechet: {avg_frechet:.4f}, Angular Cosine: {avg_angular_cosine:.4f}")
    
    return avg_loss_components

def visualize_initial_train_data(train_loader, device, config, logger):
    if not config.visualize_train_trajectories_on_start:
        return

    logger("--- Visualizing Initial Training Data ---")
    
    vis_output_dir = os.path.join(config.save_path, "train_trajectory_visualizations")
    os.makedirs(vis_output_dir, exist_ok=True)
    logger(f"Training data visualizations will be saved to: {vis_output_dir}")

    visualized_count = 0
    vis_limit = config.num_val_visualizations

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if visualized_count >= vis_limit:
                break

            try:
                point_cloud_batch = batch['point_cloud'].float().to(device)
            except KeyError as e:
                logger(f"Error: Missing key {e} in training batch {batch_idx} for visualization. Skipping batch.")
                continue
            except Exception as e:
                logger(f"Error processing training batch {batch_idx} for visualization: {e}. Skipping batch.")
                continue

            batch_size = batch['poses'].shape[0]
            samples_to_vis_in_batch = min(batch_size, vis_limit - visualized_count)

            for i in range(samples_to_vis_in_batch):
                if visualized_count >= vis_limit:
                    break

                gt_full_poses = batch['poses'][i]
                gt_full_mask = batch.get('attention_mask')[i] if batch.get('attention_mask') is not None else None
                sample_pointcloud = point_cloud_batch[i].cpu()
                
                actual_length = gt_full_poses.shape[0]
                if gt_full_mask is not None:
                    actual_length = torch.sum(gt_full_mask).int().item()
                
                if actual_length == 0:
                    logger(f"Skipping visualization for sample {i} in batch {batch_idx} due to zero actual length.")
                    continue

                history_length_for_vis = int(np.floor(actual_length * config.history_fraction))
                history_length_for_vis = max(1, min(history_length_for_vis, actual_length))


                position_dim = config.object_position_dim
                gt_full_positions = gt_full_poses[:, :position_dim]
                
                vis_past_positions = gt_full_positions[:history_length_for_vis]
                vis_future_positions_gt = gt_full_positions[history_length_for_vis:actual_length]
                
                vis_past_mask = gt_full_mask[:history_length_for_vis] if gt_full_mask is not None else None
                vis_future_mask_gt = gt_full_mask[history_length_for_vis:actual_length] if gt_full_mask is not None else None

                obj_name = batch['object_name'][i] if 'object_name' in batch else f"unknown_train_{i}"
                segment_idx = batch['segment_idx'][i].item() if 'segment_idx' in batch and batch['segment_idx'][i].item() != -1 else None
                current_sequence_path = batch['sequence_path'][i]
                sequence_name_extracted = os.path.splitext(os.path.basename(current_sequence_path))[0]

                fn_obj_name = obj_name
                fn_seq_name = f"seq_{sequence_name_extracted}"
                fn_seg_id = f"seg{segment_idx}" if segment_idx is not None else "segNA"
                fn_batch_id = f"batch{batch_idx}_sample{i}"

                filename_base = f"{fn_obj_name}_{fn_seq_name}_{fn_seg_id}_{fn_batch_id}"

                title_obj_name = obj_name
                title_seq_name = f"Seq: {sequence_name_extracted}"
                title_seg_id = f"Seg: {segment_idx if segment_idx is not None else 'NA'}"
                title_batch_id = f"Batch: {batch_idx}, Sample: {i}"
                vis_title_base = f"TRAIN - {title_obj_name} ({title_seq_name}, {title_seg_id}, {title_batch_id})"

                full_traj_path = os.path.join(vis_output_dir, f"{filename_base}_full_trajectory_with_scene.png")
                
                gt_full_positions_vis = transform_coords_for_visualization(gt_full_positions.cpu())
                sample_pointcloud_vis = transform_coords_for_visualization(sample_pointcloud.cpu())

                visualize_full_trajectory(
                    positions=gt_full_positions_vis,
                    attention_mask=gt_full_mask.cpu() if gt_full_mask is not None else None,
                    point_cloud=sample_pointcloud_vis,
                    title=f"Full GT - {vis_title_base}",
                    save_path=full_traj_path,
                    segment_idx=segment_idx
                )
                
                split_traj_path = os.path.join(vis_output_dir, f"{filename_base}_trajectory_split.png")
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
                visualized_count += 1
            
            if visualized_count >= vis_limit:
                break
    logger(f"--- Finished visualizing {visualized_count} initial training samples ---")


def main():
    # --- Configuration ---
    print("Loading configuration...")
    config = ADTObjectMotionConfig().get_configs()
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    original_save_path = config.save_path
    parent_dir = os.path.dirname(original_save_path)
    folder_name = os.path.basename(original_save_path)
    timestamped_folder_name = f"{timestamp}_{folder_name}"
    config.save_path = os.path.join(parent_dir, timestamped_folder_name)
    # TODO for dataset speed up
    config.save_path = original_save_path
    
    print(f"Original save path: {original_save_path}")
    print(f"Timestamped save path: {config.save_path}")
    
    print("Configuration loaded:")
    print(config)

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Logging Setup ---
    os.makedirs(config.save_path, exist_ok=True)
    log_file = os.path.join(config.save_path, 'train_log.txt')
    def logger(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
    logger(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger(f"Config: {vars(config)}\n")

    # --- WandB Initialization ---
    if config.wandb_mode != 'disabled':
        try:
            save_path_comment = os.path.basename(os.path.normpath(config.save_path))
            run_name = f"SimpleTransformer_ADT_{timestamp}_{save_path_comment}"
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

    train_sequences = []
    val_sequences = []
    loaded_from_files = False

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

    if train_sequences and val_sequences:
        try:
            os.makedirs(config.save_path, exist_ok=True)
            train_split_save_path = os.path.join(config.save_path, 'train_sequences.txt')
            val_split_save_path = os.path.join(config.save_path, 'val_sequences.txt')

            with open(train_split_save_path, 'w') as f:
                for seq_path in train_sequences:
                    f.write(f"{seq_path}\n")
            logger(f"Saved final training sequence list ({len(train_sequences)} sequences) to {train_split_save_path}")

            with open(val_split_save_path, 'w') as f:
                for seq_path in val_sequences:
                    f.write(f"{seq_path}\n")
            logger(f"Saved final validation sequence list ({len(val_sequences)} sequences) to {val_split_save_path}")

        except Exception as e:
             logger(f"Warning: Could not save final sequence lists: {e}")
    else:
        logger("Error: No train or validation sequences were determined. Cannot proceed.")
        return

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

    if len(train_dataset) == 0:
        logger("Error: Training dataset is empty. Check sequence paths and data."); return
    if len(val_dataset) == 0:
         logger("Error: Validation dataset is empty. Check sequence paths and data."); return

    # For the simplified model, we still need the collate_fn for data preparation
    train_collate_fn = partial(gimo_collate_fn, dataset=train_dataset, num_sample_points=config.sample_points)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True, collate_fn=train_collate_fn)
    
    val_collate_fn = partial(gimo_collate_fn, dataset=val_dataset, num_sample_points=config.sample_points)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, drop_last=True, collate_fn=val_collate_fn)
    logger(f"Train Dataset size: {len(train_dataset)}, Val Dataset size: {len(val_dataset)}, DataLoaders ready.")

    gradient_tracker = GradientTracker() if config.enable_gradient_tracking else None
    gradient_log_freq = config.gradient_log_freq if config.enable_gradient_tracking else None
    gradient_plot_freq = config.gradient_plot_freq if config.gradient_plot_freq is not None else config.val_fre
    
    if config.enable_orientation_analysis:
        print("\n=== ORIENTATION DISTRIBUTION ANALYSIS ===")
        orientation_analysis_dir = os.path.join(config.save_path, "orientation_analysis")
        os.makedirs(orientation_analysis_dir, exist_ok=True)
        
        try:
            orientation_dist_path = os.path.join(orientation_analysis_dir, "orientation_distribution_comparison.png")
            orientation_stats = plot_orientation_distribution(
                train_loader=train_loader,
                val_loader=val_loader,
                num_bins=36,
                output_path=orientation_dist_path,
                max_samples_per_loader=config.orientation_analysis_samples,
                wandb_run=wandb if config.wandb_mode != 'disabled' else None
            )
            
            stats_save_path = os.path.join(orientation_analysis_dir, "orientation_statistics.json")
            with open(stats_save_path, 'w') as f:
                json_stats = {}
                for key, value in orientation_stats.items():
                    if isinstance(value, dict):
                        json_stats[key] = {k: float(v) for k, v in value.items()}
                    else:
                        json_stats[key] = float(value) if hasattr(value, 'item') else value
                json.dump(json_stats, f, indent=2)
            logger(f"Orientation statistics saved to: {stats_save_path}")
            
        except Exception as e:
            logger(f"Warning: Could not analyze orientation distributions: {e}")
        print("=== END ORIENTATION DISTRIBUTION ANALYSIS ===\n")
    else:
        logger("Orientation distribution analysis disabled. Use --enable_orientation_analysis to enable.")

    if config.visualize_train_trajectories_on_start:
        visualize_initial_train_data(train_loader, device, config, logger)

    # --- Model Initialization ---
    logger("Initializing model...")
    logger("Using SimpleTrajectoryTransformer.")
    model = SimpleTrajectoryTransformer(config).to(device)
    
    # --- Log Model Architecture ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger("\n=== MODEL ARCHITECTURE DETAILS ===")
    logger(f"Model Type: {type(model).__name__}")
    logger(f"Total Parameters: {total_params:,}")
    logger(f"Trainable Parameters: {trainable_params:,}")
    logger(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    model_size_mb = (total_params * 4) / (1024 * 1024)
    logger(f"Estimated Model Size: {model_size_mb:.2f} MB")
    
    logger(f"\n--- Configuration Flags ---")
    logger(f"Text Embedding Enabled: {model.use_text_embedding}")
    logger(f"BPS Encoding Enabled: {getattr(model, 'use_bps', False)}")
    
    logger("\n--- Component-wise Parameter Breakdown ---")
    
    def log_component_params(component_name, component):
        if component is not None:
            component_params = sum(p.numel() for p in component.parameters())
            component_trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
            logger(f"{component_name}:")
            logger(f"  Total: {component_params:,} parameters")
            logger(f"  Trainable: {component_trainable:,} parameters")
            logger(f"  Frozen: {component_params - component_trainable:,} parameters")
            return component_params
        else:
            logger(f"{component_name}: Not used")
            return 0
    
    log_component_params("Transformer", getattr(model, 'transformer', None))
    log_component_params("Category Embedder", getattr(model, 'category_embedder', None))
    log_component_params("BPS Encoder", getattr(model, 'bps_encoder', None))
    log_component_params("Output Layer", getattr(model, 'output_layer', None))
    
    logger("=== END MODEL ARCHITECTURE DETAILS ===\n")
    
    if config.wandb_mode != 'disabled':
        wandb.watch(model)

    # --- Optimizer and Scheduler ---
    logger("Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    logger(f"Optimizer: AdamW with lr={config.lr}, beta1={config.adam_beta1}, beta2={config.adam_beta2}, eps={config.adam_eps}, weight_decay={config.weight_decay}")
    logger(f"Scheduler: ExponentialLR with gamma={config.gamma}")

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
    logger("\n--- Starting Training Loop ---")
    num_epochs = config.epoch
    
    for epoch in range(start_epoch, num_epochs + 1):
        logger(f"\nEpoch {epoch}/{num_epochs}")
        model.train() 
        epoch_total_loss = 0.0
        epoch_loss_components = {}

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Compatibility: rename 'full_poses' to 'poses' if it exists
                if 'full_poses' in batch and 'poses' not in batch:
                    batch['poses'] = batch.pop('full_poses')
                if 'full_attention_mask' in batch and 'attention_mask' not in batch:
                    batch['attention_mask'] = batch.pop('full_attention_mask')
                    
                # Move necessary tensors to device
                batch['poses'] = batch['poses'].float().to(device)
                batch['attention_mask'] = batch['attention_mask'].to(device)
                # object_category is a list of strings and stays on CPU
            except KeyError as e: logger(f"Error: Missing key {e} in batch {batch_idx}. Skipping."); continue
            except Exception as e: logger(f"Error processing batch {batch_idx}: {e}. Skipping."); continue

            # 1. Forward pass - model now takes the whole batch dictionary
            predicted_full_trajectory, cond_mask = model(batch, history_fraction=config.history_fraction, conditioning_strategy=config.conditioning_strategy, waypoint_interval=config.waypoint_interval)

            # 2. Compute loss
            total_loss, loss_dict = model.compute_loss(predicted_full_trajectory, cond_mask, batch)

            # 3. Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            
            if gradient_tracker is not None and batch_idx % gradient_log_freq == 0:
                gradient_tracker.log_gradients(model, loss_dict)
            
            optimizer.step()

            epoch_total_loss += total_loss.item()
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    epoch_loss_components[key] = epoch_loss_components.get(key, 0.0) + value.item()

            progress_bar.set_postfix({'loss': f"{total_loss.item():.4f}"})

        avg_train_loss = epoch_total_loss / len(train_loader)
        avg_train_components = {k: v / len(train_loader) for k, v in epoch_loss_components.items()}
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
                    if epoch == config.val_fre:
                        trajectory_vis_dir = os.path.join(config.save_path, "trajectory_visualizations")
                        if os.path.exists(trajectory_vis_dir):
                            try:
                                traj_images = [img for img in os.listdir(trajectory_vis_dir) 
                                             if img.endswith('.png')]
                                
                                object_traj_images = {}
                                for img in traj_images:
                                    object_name = img.split('_')[0] if '_' in img else None
                                    if object_name:
                                        if object_name not in object_traj_images:
                                            object_traj_images[object_name] = []
                                        object_traj_images[object_name].append(img)
                                
                                first_object_traj_images = []
                                if object_traj_images:
                                    first_object = sorted(object_traj_images.keys())[0]
                                    first_object_traj_images = object_traj_images[first_object]
                                
                                if first_object_traj_images:
                                    wandb.log({"trajectory_visualizations": [wandb.Image(os.path.join(trajectory_vis_dir, img)) 
                                                                           for img in first_object_traj_images]}, step=epoch)
                            except Exception as e:
                                logger(f"Warning: Failed to log trajectory visualizations to WandB: {e}")
                    
                    vis_output_dir = os.path.join(config.save_path, "val_visualizations", f"epoch_{epoch}")
                    if os.path.exists(vis_output_dir):
                        try:
                            pred_images = [img for img in os.listdir(vis_output_dir) 
                                           if img.endswith('.png') and 'prediction_vs_gt' in img]
                            
                            object_pred_images = {}
                            for img in pred_images:
                                object_name = img.split('_')[0] if '_' in img else None
                                if object_name:
                                    if object_name not in object_pred_images:
                                        object_pred_images[object_name] = []
                                    object_pred_images[object_name].append(img)
                            
                            first_object_pred_images = []
                            if object_pred_images:
                                first_object = sorted(object_pred_images.keys())[0]
                                first_object_pred_images = object_pred_images[first_object]
                            
                            if first_object_pred_images:
                                wandb.log({"prediction_visualizations": [wandb.Image(os.path.join(vis_output_dir, img)) 
                                                                       for img in first_object_pred_images]}, step=epoch)
                        except Exception as e:
                            logger(f"Warning: Failed to log prediction visualizations to WandB: {e}")

            current_val_loss = val_metrics['total_loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                logger(f"---> New best validation loss: {best_val_loss:.4f}. Saving best model...")
                best_model_path = os.path.join(config.save_path, 'best_model.pth')
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

        if epoch % gradient_plot_freq == 0 and gradient_tracker is not None and len(gradient_tracker.get_logs()) > 0:
            logger("=== GRADIENT ANALYSIS ===")
            try:
                gradient_analysis_dir = os.path.join(config.save_path, "gradient_analysis")
                os.makedirs(gradient_analysis_dir, exist_ok=True)
                
                gradient_log_path = os.path.join(gradient_analysis_dir, f"gradient_logs_epoch_{epoch}.pth")
                gradient_tracker.save_logs(gradient_log_path)
                
                gradient_plot_path = os.path.join(gradient_analysis_dir, f"gradient_magnitudes_epoch_{epoch}.png")
                gradient_stats = plot_gradient_magnitudes(
                    gradient_logs=gradient_tracker.get_logs(),
                    output_path=gradient_plot_path,
                    window_size=20,
                    show_losses=True,
                    wandb_run=wandb if config.wandb_mode != 'disabled' else None,
                    step=epoch
                )
                
                gradient_stats_path = os.path.join(gradient_analysis_dir, f"gradient_statistics_epoch_{epoch}.json")
                with open(gradient_stats_path, 'w') as f:
                    json.dump(gradient_stats, f, indent=2)
                
                if config.wandb_mode != 'disabled':
                    try:
                        wandb.log({"gradient_analysis": wandb.Image(gradient_plot_path)}, step=epoch)
                        for component, stats in gradient_stats.items():
                            wandb.log({f"grad_stats/{component}_mean": stats['mean']}, step=epoch)
                            wandb.log({f"grad_stats/{component}_final": stats['final']}, step=epoch)
                    except Exception as e:
                        logger(f"Warning: Failed to log gradient analysis to WandB: {e}")
                
                logger(f"Gradient analysis completed for epoch {epoch}")
            except Exception as e:
                logger(f"Warning: Could not perform gradient analysis: {e}")

        if epoch % config.save_fre == 0:
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

        scheduler.step()

    logger("\n--- Training Finished ---")
    
    if gradient_tracker is not None and len(gradient_tracker.get_logs()) > 0:
        logger("=== FINAL GRADIENT ANALYSIS ===")
        try:
            gradient_analysis_dir = os.path.join(config.save_path, "gradient_analysis")
            os.makedirs(gradient_analysis_dir, exist_ok=True)
            
            final_gradient_log_path = os.path.join(gradient_analysis_dir, "final_gradient_logs.pth")
            gradient_tracker.save_logs(final_gradient_log_path)
            
            final_gradient_plot_path = os.path.join(gradient_analysis_dir, "final_gradient_magnitudes.png")
            final_gradient_stats = plot_gradient_magnitudes(
                gradient_logs=gradient_tracker.get_logs(),
                output_path=final_gradient_plot_path,
                window_size=20,
                show_losses=True,
                wandb_run=wandb if config.wandb_mode != 'disabled' else None,
                step=num_epochs
            )
            
            final_gradient_stats_path = os.path.join(gradient_analysis_dir, "final_gradient_statistics.json")
            with open(final_gradient_stats_path, 'w') as f:
                json.dump(final_gradient_stats, f, indent=2)
            
            logger("Final gradient analysis completed")
            
            logger("\n=== GRADIENT ANALYSIS SUMMARY ===")
            for component, stats in final_gradient_stats.items():
                logger(f"{component}: Mean={stats['mean']:.6f}, Final={stats['final']:.6f}, Trend={stats['trend']}")
            
        except Exception as e:
            logger(f"Warning: Could not perform final gradient analysis: {e}")
    
    final_model_path = os.path.join(config.save_path, 'final_model.pth')
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

if __name__ == '__main__':
    main() 