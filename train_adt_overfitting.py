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
    # Make vis_output_dir specific to overfitting run
    vis_output_dir = os.path.join(config.save_path, "overfitting_val_visualizations", f"epoch_{epoch}")
    if vis_limit > 0:
        os.makedirs(vis_output_dir, exist_ok=True)
        print(f"Validation visualizations will be saved to: {vis_output_dir}")

    print("\nRunning validation...")
    with torch.no_grad():
        # Since batch size is 1, no need for progress bar usually
        for batch_idx, batch in enumerate(dataloader):
            try:
                full_trajectory_batch = batch['full_poses'].float().to(device)
                point_cloud_batch = batch['point_cloud'].float().to(device) # Get point cloud from collated batch
                # Move mask to device for loss calc
                batch['full_attention_mask'] = batch['full_attention_mask'].to(device)
                
                # Prepare input trajectory for the model based on config
                if config.use_first_frame_only:
                    # Use only the first frame as input
                    input_trajectory_batch = full_trajectory_batch[:, 0:1, :]
                else:
                    # Use a fixed portion of the history as input
                    fixed_history_length = int(np.floor(full_trajectory_batch.shape[1] * config.history_fraction))
                    input_trajectory_batch = full_trajectory_batch[:, :fixed_history_length, :]
                
                # Ensure object names are usable (e.g., list of strings)
                object_names = batch.get('object_name', [f"unknown_{i}" for i in range(full_trajectory_batch.shape[0])])
                # Ensure object IDs are usable
                object_ids = batch.get('object_id', torch.arange(full_trajectory_batch.shape[0])).cpu().numpy()

            except KeyError as e:
                print(f"Error: Missing key {e} in validation batch {batch_idx}. Skipping batch.")
                continue
            except Exception as e:
                print(f"Error processing validation batch {batch_idx}: {e}. Skipping batch.")
                continue

            # Forward pass with input trajectory only
            predicted_full_trajectory = model(input_trajectory_batch, point_cloud_batch)
            total_loss, loss_dict = model.compute_loss(predicted_full_trajectory, batch)

            val_total_loss += total_loss.item()
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    val_loss_components[key] = val_loss_components.get(key, 0.0) + value.item()

            # Optional: Log validation loss per step if desired for overfitting
            # print(f"Validation step loss: {total_loss.item():.4f}")

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
                    gt_full_orientations = gt_full_poses_batch[i, :, 3:]
                    gt_full_mask = gt_full_mask_batch[i] if gt_full_mask_batch is not None else None
                    pred_full_positions = pred_full_trajectory_batch[i, :, :3]
                    pred_full_orientations = pred_full_trajectory_batch[i, :, 3:]
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
                    
                    vis_past_orientations = gt_full_orientations[:history_length_for_vis]
                    vis_future_orientations_gt = gt_full_orientations[history_length_for_vis:actual_length]

                    vis_past_mask = gt_full_mask[:history_length_for_vis] if gt_full_mask is not None else None
                    vis_future_mask_gt = gt_full_mask[history_length_for_vis:actual_length] if gt_full_mask is not None else None

                    # Slice prediction dynamically based on use_first_frame_only
                    if config.use_first_frame_only:
                        valid_future_len = actual_length - history_length_for_vis
                        predicted_future_positions = pred_full_positions[:valid_future_len] 
                        predicted_future_orientations = pred_full_orientations[:valid_future_len]
                    else:
                        # Standard case: model output is full trajectory
                        pred_past_positions = pred_full_positions[:history_length_for_vis]
                        predicted_future_positions = pred_full_positions[history_length_for_vis:actual_length]
                        
                        pred_past_orientations = pred_full_orientations[:history_length_for_vis]
                        predicted_future_orientations = pred_full_orientations[history_length_for_vis:actual_length]
                    # ----------------------------------------------------

                    obj_name = object_names[i]
                    segment_idx = batch['segment_idx'][i].item() if 'segment_idx' in batch and batch['segment_idx'][i].item() != -1 else None
                    
                    if segment_idx is not None:
                        filename_base = f"{obj_name}_seg{segment_idx}"
                        vis_title_base = f"{obj_name} (Seg: {segment_idx})"
                    else:
                        filename_base = f"{obj_name}"
                        vis_title_base = f"{obj_name}"
                    
                    # Check if orientation visualization is enabled
                    show_ori_arrows = getattr(config, 'show_ori_arrows', False)
                    viz_ori_scale = getattr(config, 'viz_ori_scale', 0.2)
                    
                    # Full Trajectory Visualization - now with point cloud and orientation
                    full_traj_path = os.path.join(vis_output_dir, f"{filename_base}_full_trajectory_with_scene_epoch{epoch}.png")
                    visualize_full_trajectory(
                        positions=gt_full_positions,
                        attention_mask=gt_full_mask,
                        point_cloud=sample_pointcloud,  # Pass the point cloud
                        title=f"Full GT - {vis_title_base}",
                        save_path=full_traj_path,
                        segment_idx=segment_idx,
                        show_orientation=show_ori_arrows,
                        orientations=gt_full_orientations,
                        arrow_scale=viz_ori_scale
                    )
                    
                    # Split Trajectory Visualization (uses dynamically sliced data)
                    split_traj_path = os.path.join(vis_output_dir, f"{filename_base}_trajectory_split.png")
                    visualize_trajectory(
                        past_positions=vis_past_positions,
                        future_positions=vis_future_positions_gt,
                        past_mask=vis_past_mask,
                        future_mask=vis_future_mask_gt,
                        title=f"Split GT - {vis_title_base}",
                        save_path=split_traj_path,
                        segment_idx=segment_idx,
                        show_orientation=show_ori_arrows,
                        past_orientations=vis_past_orientations,
                        future_orientations=vis_future_orientations_gt,
                        arrow_scale=viz_ori_scale
                    )
                    
                    # Prediction vs GT Visualization (uses dynamically sliced data)
                    pred_vs_gt_path = os.path.join(vis_output_dir, f"{filename_base}_prediction_vs_gt_epoch{epoch}.png")
                    visualize_prediction(
                        past_positions=vis_past_positions,
                        future_positions_gt=vis_future_positions_gt,
                        future_positions_pred=predicted_future_positions, # Use dynamically sliced prediction
                        past_mask=vis_past_mask,
                        future_mask_gt=vis_future_mask_gt, # Corrected argument name
                        title=f"Pred vs GT - {vis_title_base} (Epoch {epoch})",
                        save_path=pred_vs_gt_path,
                        segment_idx=segment_idx,
                        show_orientation=show_ori_arrows,
                        past_orientations=vis_past_orientations,
                        future_orientations_gt=vis_future_orientations_gt,
                        future_orientations_pred=predicted_future_orientations,
                        arrow_scale=viz_ori_scale
                    )
                    
                    visualized_count += 1
            # --- End Visualization Logic ---

    # Average loss calculation is simpler since len(dataloader) is 1
    avg_val_loss = val_total_loss
    avg_loss_components = val_loss_components # No division needed
    avg_loss_components['total_loss'] = avg_val_loss
    
    return avg_loss_components

# --- Custom Collate Function (Updated for trajectory-specific point clouds) ---
def gimo_collate_fn(batch, dataset, num_sample_points):
    """
    Custom collate function to handle trajectory-specific point clouds.
    
    Args:
        batch (list): A list of sample dictionaries (will contain only one dict in overfitting mode).
        dataset (GIMOMultiSequenceDataset): The instance of the original dataset being used.
        num_sample_points (int): The number of points to sample from each point cloud.

    Returns:
        dict: A batch dictionary suitable for the model, including batched point clouds.
    """
    # Create a list to store point clouds for each item in the batch
    point_cloud_batch_list = []
    
    for i, item in enumerate(batch):
        # First check if the item has its own trajectory-specific point cloud
        if 'trajectory_specific_pointcloud' in item and item['trajectory_specific_pointcloud'] is not None:
            point_cloud = item['trajectory_specific_pointcloud']
            
            # Convert to tensor if it's a numpy array
            if isinstance(point_cloud, np.ndarray):
                point_cloud = torch.from_numpy(point_cloud).float()
                
            print(f"Using trajectory-specific point cloud with {point_cloud.shape[0]} points")
        else:
            # Fallback to getting from dataset if not included in the item
            dataset_idx = item.get('dataset_idx', 0)
            point_cloud = dataset.get_scene_pointcloud(dataset_idx)
            
            if point_cloud is None:
                print(f"Warning: Failed to get point cloud for dataset_idx {dataset_idx} in batch. Using zeros.")
                # Create a dummy point cloud if loading fails
                point_cloud = torch.zeros((num_sample_points, 3), dtype=torch.float32)
            elif isinstance(point_cloud, np.ndarray):
                point_cloud = torch.from_numpy(point_cloud).float()
            
            print(f"Using full scene point cloud with {point_cloud.shape[0]} points")
        
        # Sample the point cloud to ensure consistent size
        if point_cloud.shape[0] >= num_sample_points:
            # Randomly sample points without replacement
            indices = np.random.choice(point_cloud.shape[0], num_sample_points, replace=False)
            sampled_pc = point_cloud[indices]
        else:
            # If not enough points, sample with replacement
            print(f"Warning: Point cloud has only {point_cloud.shape[0]} points. Sampling with replacement to get {num_sample_points}.")
            indices = np.random.choice(point_cloud.shape[0], num_sample_points, replace=True)
            sampled_pc = point_cloud[indices]
        
        # Add to batch list
        point_cloud_batch_list.append(sampled_pc)
    
    # Stack the point clouds
    batched_point_clouds = torch.stack(point_cloud_batch_list, dim=0)
    
    # Remove trajectory-specific point clouds and dataset_idx from items to avoid issues with default_collate
    batch_copy = []
    for item in batch:
        item_copy = {k: v for k, v in item.items() if k not in ['trajectory_specific_pointcloud', 'dataset_idx']}
        batch_copy.append(item_copy)
    
    # Collate the rest using default_collate
    collated_batch = default_collate(batch_copy)
    
    # Add the batched point clouds
    collated_batch['point_cloud'] = batched_point_clouds
    
    return collated_batch
# ---------------------------

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
            comment_suffix = f"_overfit_{config.comment}" if config.comment else "_overfit"
            run_name = f"GIMO_ADT_{time.strftime('%Y%m%d_%H%M%S')}{comment_suffix}" # Add overfit tag
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
    first_sample = original_train_dataset[0]
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
    model = GIMO_ADT_Model(config).to(device)
    logger(f"Model initialized. Parameter count: {sum(p.numel() for p in model.parameters())}")
    if config.wandb_mode != 'disabled':
        wandb.watch(model)

    # --- Optimizer and Scheduler --- 
    logger("Setting up optimizer and scheduler...")
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    logger(f"Optimizer: Adam with lr={config.lr}, weight_decay={config.weight_decay}")
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
                full_trajectory_batch = batch['full_poses'].float().to(device)
                point_cloud_batch = batch['point_cloud'].float().to(device)
                batch['full_attention_mask'] = batch['full_attention_mask'].to(device)
                
                # Prepare input trajectory for the model based on config
                if config.use_first_frame_only:
                    # Use only the first frame as input
                    input_trajectory_batch = full_trajectory_batch[:, 0:1, :]
                else:
                    # Use a fixed portion of the history as input
                    fixed_history_length = int(np.floor(full_trajectory_batch.shape[1] * config.history_fraction))
                    input_trajectory_batch = full_trajectory_batch[:, :fixed_history_length, :]

            except KeyError as e: logger(f"Error: Missing key {e} in batch {batch_idx}. Skipping."); continue
            except Exception as e: logger(f"Error processing batch {batch_idx}: {e}. Skipping."); continue

            # Forward pass with input trajectory only
            predicted_full_trajectory = model(input_trajectory_batch, point_cloud_batch)
            total_loss, loss_dict = model.compute_loss(predicted_full_trajectory, batch)

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
                    vis_output_dir = os.path.join(config.save_path, "overfitting_val_visualizations", f"epoch_{epoch}")
                    if os.path.exists(vis_output_dir):
                         try:
                             all_images = [img for img in os.listdir(vis_output_dir) 
                                          if img.endswith('.png') and 'epoch' in img]
                             # Since it's only one object, log all its images
                             if all_images:
                                 wandb.log({"val_visualizations": [wandb.Image(os.path.join(vis_output_dir, img)) 
                                                                  for img in sorted(all_images)]}, step=epoch)
                         except Exception as e:
                              logger(f"Warning: Failed to log validation images to WandB: {e}")

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