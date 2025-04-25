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

# Imports from our project
from config.adt_config import ADTObjectMotionConfig
from dataset.gimo_multi_sequence_dataset import GIMOMultiSequenceDataset
from model.gimo_adt_model import GIMO_ADT_Model
from torch.utils.data import DataLoader
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
    vis_output_dir = os.path.join(config.save_path, "val_visualizations", f"epoch_{epoch}")
    if vis_limit > 0:
        os.makedirs(vis_output_dir, exist_ok=True)
        print(f"Validation visualizations will be saved to: {vis_output_dir}")

    print("\nRunning validation...")
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                full_trajectory_batch = batch['full_positions'].float().to(device)
                point_cloud_batch = batch['point_cloud'].float().to(device) # Get point cloud from collated batch
                # Move mask to device for loss calc
                batch['full_attention_mask'] = batch['full_attention_mask'].to(device)
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

            predicted_full_trajectory = model(full_trajectory_batch, point_cloud_batch)
            total_loss, loss_dict = model.compute_loss(predicted_full_trajectory, batch)

            val_total_loss += total_loss.item()
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    val_loss_components[key] = val_loss_components.get(key, 0.0) + value.item()

            progress_bar.set_postfix({'val_loss': f"{total_loss.item():.4f}"})

            # --- Visualization Logic ---
            if visualized_count < vis_limit:
                batch_size = full_trajectory_batch.shape[0]
                samples_to_vis = min(batch_size, vis_limit - visualized_count)
                
                # Extract data for visualization
                gt_full_positions_batch = batch['full_positions'] # Renamed for clarity
                gt_full_mask_batch = batch.get('full_attention_mask')
                pred_full_trajectory_batch = predicted_full_trajectory # Renamed for clarity
                
                # No need to slice based on fixed model lengths here anymore
                # hist_len = model.history_length
                # fut_len = model.future_length
                # vis_past_positions = gt_full_positions[:, :hist_len, :]
                # vis_future_positions_gt = gt_full_positions[:, hist_len:hist_len+fut_len, :]
                # vis_past_mask = gt_full_mask[:, :hist_len] if gt_full_mask is not None else None
                # vis_future_mask_gt = gt_full_mask[:, hist_len:hist_len+fut_len] if gt_full_mask is not None else None
                # predicted_future_vis = predicted_full_trajectory[:, -fut_len:, :]
                
                for i in range(samples_to_vis):
                    if visualized_count >= vis_limit:
                        break # Ensure we don't exceed limit within the inner loop
                        
                    # Get data for the current sample
                    gt_full_positions = gt_full_positions_batch[i]
                    gt_full_mask = gt_full_mask_batch[i] if gt_full_mask_batch is not None else None
                    pred_full_trajectory = pred_full_trajectory_batch[i]
                    
                    # --- Dynamic Split Calculation for Visualization ---
                    if gt_full_mask is None:
                        print("Warning: Cannot perform dynamic split for visualization - mask missing.")
                        # Fallback or skip visualization for this sample
                        actual_length = gt_full_positions.shape[0]
                        if config.use_first_frame_only:
                            history_length_for_vis = 1
                        else:
                            history_length_for_vis = int(np.floor(actual_length * config.history_fraction))
                            history_length_for_vis = max(1, min(history_length_for_vis, actual_length)) # Ensure valid length
                    else:
                        actual_length = torch.sum(gt_full_mask).int().item()
                        if config.use_first_frame_only:
                            history_length_for_vis = 1 if actual_length >= 1 else 0 # Past is frame 0
                        else:
                            history_length_for_vis = int(np.floor(actual_length * config.history_fraction))
                            history_length_for_vis = max(1, min(history_length_for_vis, actual_length)) # Ensure valid length

                    # Slice GT based on the calculated history length for visualization
                    vis_past_positions = gt_full_positions[:history_length_for_vis]
                    vis_future_positions_gt = gt_full_positions[history_length_for_vis:actual_length] # Slice up to actual length

                    vis_past_mask = gt_full_mask[:history_length_for_vis] if gt_full_mask is not None else None
                    vis_future_mask_gt = gt_full_mask[history_length_for_vis:actual_length] if gt_full_mask is not None else None

                    # Slice prediction dynamically (relative to prediction length which should match GT length)
                    pred_past_vis = pred_full_trajectory[:history_length_for_vis]
                    predicted_future_vis = pred_full_trajectory[history_length_for_vis:actual_length]
                    # ----------------------------------------------------

                    obj_name = object_names[i]
                    # Get segment_idx from batch if available
                    segment_idx = batch['segment_idx'][i].item() if 'segment_idx' in batch and batch['segment_idx'][i].item() != -1 else None
                    
                    # Create filename base
                    if segment_idx is not None:
                        filename_base = f"{obj_name}_seg{segment_idx}"
                        vis_title_base = f"{obj_name} (Seg: {segment_idx})"
                    else:
                        filename_base = f"{obj_name}"
                        vis_title_base = f"{obj_name}"
                    
                    # Full Trajectory Visualization (uses full data)
                    full_traj_path = os.path.join(vis_output_dir, f"{filename_base}_full_trajectory.png")
                    visualize_full_trajectory(
                        positions=gt_full_positions,
                        attention_mask=gt_full_mask,
                        title=f"Full GT - {vis_title_base}",
                        save_path=full_traj_path,
                        segment_idx=segment_idx
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
                        segment_idx=segment_idx
                    )
                    
                    # Prediction vs GT Visualization (uses dynamically sliced data)
                    pred_vs_gt_path = os.path.join(vis_output_dir, f"{filename_base}_prediction_vs_gt_epoch{epoch}.png")
                    visualize_prediction(
                        past_positions=vis_past_positions,
                        future_positions_gt=vis_future_positions_gt,
                        future_positions_pred=predicted_future_vis, # Use dynamically sliced prediction
                        past_mask=vis_past_mask,
                        future_mask_gt=vis_future_mask_gt,
                        title=f"Pred vs GT - {vis_title_base} (Epoch {epoch})",
                        save_path=pred_vs_gt_path,
                        segment_idx=segment_idx
                    )
                    
                    visualized_count += 1
            # --- End Visualization Logic ---

    avg_val_loss = val_total_loss / len(dataloader)
    avg_loss_components = {k: v / len(dataloader) for k, v in val_loss_components.items()}
    avg_loss_components['total_loss'] = avg_val_loss
    
    return avg_loss_components

# --- Custom Collate Function ---
def gimo_collate_fn(batch, dataset, num_sample_points):
    """
    Custom collate function to handle varying point clouds.
    
    Args:
        batch (list): A list of sample dictionaries from GIMOMultiSequenceDataset.
                      Each dict must contain 'dataset_idx' and other required data.
        dataset (GIMOMultiSequenceDataset): The instance of the dataset being used.
                                            Needed to access get_scene_pointcloud.
        num_sample_points (int): The number of points to sample from each point cloud.

    Returns:
        dict: A batch dictionary suitable for the model, including batched point clouds.
    """
    # Separate point clouds based on dataset_idx
    pc_dict = {}
    batch_dataset_indices = [item['dataset_idx'] for item in batch]
    
    for i, dataset_idx in enumerate(batch_dataset_indices):
        if dataset_idx not in pc_dict:
            # Load point cloud if not already loaded for this batch
            point_cloud = dataset.get_scene_pointcloud(dataset_idx)
            if point_cloud is None:
                print(f"Warning: Failed to get point cloud for dataset_idx {dataset_idx} in batch. Using zeros.")
                # Create a dummy point cloud if loading fails
                point_cloud = torch.zeros((num_sample_points, 3), dtype=torch.float32)
            elif isinstance(point_cloud, np.ndarray):
                point_cloud = torch.from_numpy(point_cloud).float()
            
            # Ensure point cloud has enough points, sample if necessary
            if point_cloud.shape[0] >= num_sample_points:
                # Randomly sample points
                indices = np.random.choice(point_cloud.shape[0], num_sample_points, replace=False)
                sampled_pc = point_cloud[indices]
            else:
                # If not enough points, sample with replacement (or pad, but sampling is simpler)
                print(f"Warning: Point cloud for dataset_idx {dataset_idx} has only {point_cloud.shape[0]} points. Sampling with replacement to get {num_sample_points}.")
                indices = np.random.choice(point_cloud.shape[0], num_sample_points, replace=True)
                sampled_pc = point_cloud[indices]
                
            pc_dict[dataset_idx] = sampled_pc
            
    # Create the point cloud batch tensor
    point_cloud_batch_list = [pc_dict[idx] for idx in batch_dataset_indices]
    batched_point_clouds = torch.stack(point_cloud_batch_list, dim=0)

    # Collate other data using default_collate, excluding dataset_idx and point_cloud if present
    # Create a new list of dicts without the dataset_idx key for default collation
    batch_copy = [{k: v for k, v in item.items() if k != 'dataset_idx'} for item in batch]
    collated_batch = default_collate(batch_copy)
    
    # Add the processed point cloud batch
    collated_batch['point_cloud'] = batched_point_clouds
    
    return collated_batch
# ---------------------------

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
            comment_suffix = f"_{config.comment}" if config.comment else ""
            run_name = f"GIMO_ADT_{time.strftime('%Y%m%d_%H%M%S')}{comment_suffix}"
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
                    write_to_file=False # Don't write split files here
                )
                logger(f"Using {len(train_sequences)} sequences for training and {len(val_sequences)} for validation (dynamically split).")

                # --- Handle train_ratio = 1.0 case ---
                if config.train_ratio >= 1.0:
                     logger("Train ratio is >= 1.0, using all sequences for both training and validation.")
                     val_sequences = train_sequences
                # ---------------------------------------

            except Exception as e:
                logger(f"Error during sequence finding/splitting: {e}. Please check adt_dataroot and utils.")
                return
        elif os.path.isdir(config.adt_dataroot) and not HAS_SEQ_UTILS:
             logger("Warning: adt_sequence_utils not found. Assuming adt_dataroot contains only training sequences.")
             all_items = [os.path.join(config.adt_dataroot, item) for item in os.listdir(config.adt_dataroot)]
             train_sequences = [item for item in all_items if os.path.isdir(item)]
             val_sequences = train_sequences # Use training data for validation as fallback
             if not train_sequences:
                 logger(f"Error: No sequence directories found in {config.adt_dataroot} for training."); return
             logger(f"Using {len(train_sequences)} sequences for training and validation (from directory scan).")
        elif os.path.exists(config.adt_dataroot): # If it's a single file/sequence path
             logger(f"Using single sequence {config.adt_dataroot} for training and validation.")
             train_sequences = [config.adt_dataroot]
             val_sequences = [config.adt_dataroot]
        else:
             # This case should be caught by the initial check, but included for completeness
             logger(f"Error: Invalid adt_dataroot path {config.adt_dataroot}.")
             return

    # --- Save the final train and validation sequence lists ---
    if train_sequences and val_sequences:
        try:
            os.makedirs(config.save_path, exist_ok=True) # Ensure directory exists
            train_split_save_path = os.path.join(config.save_path, 'train_sequences.txt')
            val_split_save_path = os.path.join(config.save_path, 'val_sequences.txt')

            with open(train_split_save_path, 'w') as f:
                for seq_path in train_sequences:
                    f.write(f"{seq_path}\\n")
            logger(f"Saved final training sequence list ({len(train_sequences)} sequences) to {train_split_save_path}")

            with open(val_split_save_path, 'w') as f:
                for seq_path in val_sequences:
                    f.write(f"{seq_path}\\n")
            logger(f"Saved final validation sequence list ({len(val_sequences)} sequences) to {val_split_save_path}")

        except Exception as e:
             logger(f"Warning: Could not save final sequence lists: {e}")
    else:
        logger("Error: No train or validation sequences were determined. Cannot proceed.")
        return
    # -------------------------------------------------------

    # Create datasets
    cache_dir = os.path.join(config.save_path, 'trajectory_cache') # Use a dedicated cache dir for this run
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    
    train_dataset = GIMOMultiSequenceDataset(
        sequence_paths=train_sequences,
        config=config,  # Pass the config object
        cache_dir=cache_dir  # Still explicitly set cache_dir
    )
    val_dataset = GIMOMultiSequenceDataset(
        sequence_paths=val_sequences,
        config=config,  # Pass the config object
        cache_dir=cache_dir  # Still explicitly set cache_dir 
    )

    if len(train_dataset) == 0:
        logger("Error: Training dataset is empty. Check sequence paths and data."); return
    if len(val_dataset) == 0:
         logger("Error: Validation dataset is empty. Check sequence paths and data."); return

    # Create DataLoaders
    # Bind dataset instance and sample points count to the collate function
    collate_func = partial(gimo_collate_fn, dataset=train_dataset, num_sample_points=config.sample_points)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True, collate_fn=collate_func)
    
    # Use the same collate function structure for validation loader
    val_collate_func = partial(gimo_collate_fn, dataset=val_dataset, num_sample_points=config.sample_points)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, drop_last=True, collate_fn=val_collate_func)
    logger(f"Train Dataset size: {len(train_dataset)}, Val Dataset size: {len(val_dataset)}, DataLoaders ready with custom collate_fn.") # Updated log

    # --- Model Initialization ---
    logger("Initializing model...")
    model = GIMO_ADT_Model(config).to(device)
    logger(f"Model initialized. Parameter count: {sum(p.numel() for p in model.parameters())}")
    if config.wandb_mode != 'disabled':
        wandb.watch(model) # Watch model gradients

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
            # Optionally load optimizer, scheduler, epoch, best_val_loss
            if 'optimizer_state_dict' in checkpoint and config.load_optim_dir is None: # Prioritize specific optim path if given
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
    # Load optimizer separately if specified (overrides checkpoint)
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
                # Data is already collated, move relevant tensors to device
                full_trajectory_batch = batch['full_positions'].float().to(device)
                point_cloud_batch = batch['point_cloud'].float().to(device) # Get point cloud from collated batch
                # Move other tensors needed for loss calculation (e.g., mask)
                batch['full_attention_mask'] = batch['full_attention_mask'].to(device)

            except KeyError as e: logger(f"Error: Missing key {e} in batch {batch_idx}. Skipping."); continue
            except Exception as e: logger(f"Error processing batch {batch_idx}: {e}. Skipping."); continue

            # 1. Forward pass
            predicted_full_trajectory = model(full_trajectory_batch, point_cloud_batch)

            # 2. Compute loss
            total_loss, loss_dict = model.compute_loss(predicted_full_trajectory, batch)

            # 3. Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            epoch_total_loss += total_loss.item()
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    epoch_loss_components[key] = epoch_loss_components.get(key, 0.0) + value.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': f"{total_loss.item():.4f}"})

        # Calculate and log average epoch loss
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
                # Log visualizations to WandB if enabled
                if config.num_val_visualizations > 0:
                    vis_output_dir = os.path.join(config.save_path, "val_visualizations", f"epoch_{epoch}")
                    if os.path.exists(vis_output_dir):
                         try:
                             all_images = [img for img in os.listdir(vis_output_dir) 
                                          if img.endswith('.png') and 'epoch' in img]
                             # Sort files to ensure consistent ordering
                             all_images.sort()
                             # Group images by object name
                             object_images = {}
                             for img in all_images:
                                 object_name = img.split('_')[0] if '_' in img else None
                                 if object_name:
                                     if object_name not in object_images:
                                         object_images[object_name] = []
                                     object_images[object_name].append(img)
                             # Get images for the first object only (if any objects found)
                             first_object_images = []
                             if object_images:
                                 # Get the first object name (after sorting for consistency)
                                 first_object = sorted(object_images.keys())[0]
                                 first_object_images = object_images[first_object]
                             
                             # Only keep images for the first object found
                             if first_object_images:
                                 wandb.log({"val_visualizations": [wandb.Image(os.path.join(vis_output_dir, img)) 
                                                                  for img in first_object_images]}, step=epoch)
                         except Exception as e:
                              logger(f"Warning: Failed to log validation images to WandB: {e}")

            # Save best model based on validation loss
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
                    'config': vars(config) # Save config with model
                }, best_model_path)
                if config.wandb_mode != 'disabled':
                     wandb.save(best_model_path) # Save best model to wandb

        # --- Periodic Checkpoint Saving --- 
        if epoch % config.save_fre == 0:
            ckpt_path = os.path.join(config.save_path, f'ckpt_epoch_{epoch}.pth')
            logger(f"Saving periodic checkpoint to {ckpt_path}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss, # Save current best loss
                'config': vars(config)
            }, ckpt_path)

        # --- LR Step --- 
        scheduler.step()
        # logger(f"LR scheduler step taken. New LR: {scheduler.get_last_lr()[0]:.6f}")

    logger("\n--- Training Finished ---")
    # Save final model
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