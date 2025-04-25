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
        default=10, # Number of trajectory samples to visualize
        help="Number of trajectory samples to visualize"
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
        "--comment",
        type=str,
        default="",
        help="Optional comment for the evaluation run"
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
    
    return model, config

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

def evaluate(model, config, args):
    """
    Run the evaluation loop.
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
    eval_cache_dir = os.path.join(args.output_dir, 'trajectory_cache_eval')
    os.makedirs(eval_cache_dir, exist_ok=True)
    print(f"Using evaluation cache directory: {eval_cache_dir}")

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
                full_trajectory_batch = batch['full_positions'].float().to(device)
                point_cloud_batch = batch['point_cloud'].float().to(device)
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

                # --- Model Inference ---
                predicted_full_trajectory = model(full_trajectory_batch, point_cloud_batch)

                # --- Process Each Sample in Batch ---
                batch_l1 = 0.0
                batch_rmse_ade = 0.0
                batch_fde = 0.0
                valid_samples_in_batch = 0

                for i in range(full_trajectory_batch.shape[0]):
                    gt_full = full_trajectory_batch[i]
                    pred_full = predicted_full_trajectory[i]
                    mask_full = full_attention_mask[i]

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
                    else:
                        gt_future_denorm = gt_future
                        pred_future_denorm = pred_future

                    # Compute Metrics for this sample
                    l1_mean, rmse_ade, fde = compute_metrics_for_sample(pred_future_denorm, gt_future_denorm, future_mask)

                    batch_l1 += l1_mean.item()
                    batch_rmse_ade += rmse_ade.item()
                    batch_fde += fde.item()
                    valid_samples_in_batch += 1

                    # --- Visualization ---
                    if args.visualize:
                         try:
                             # Denormalize history as well for visualization
                             if is_normalized:
                                 gt_hist_denorm = denormalize_trajectory(gt_hist, sample_norm_params)
                                 # pred_hist_denorm = denormalize_trajectory(pred_hist, sample_norm_params) # Not needed for pred viz usually
                             else:
                                 gt_hist_denorm = gt_hist
                                 # pred_hist_denorm = pred_hist

                             obj_name = object_names[i] if i < len(object_names) else f"obj_{batch_idx}_{i}"
                             seg_idx = segment_indices[i].item() if i < len(segment_indices) and segment_indices[i].item() != -1 else None

                             if seg_idx is not None:
                                 filename_base = f"{obj_name}_seg{seg_idx}"
                                 vis_title_base = f"{obj_name} (Seg: {seg_idx})"
                             else:
                                 filename_base = f"{obj_name}"
                                 vis_title_base = f"{obj_name}"
                                 
                             pred_vs_gt_path = os.path.join(vis_output_dir, f"{filename_base}_prediction_vs_gt.png")

                             visualize_prediction(
                                 past_positions=gt_hist_denorm.cpu(),
                                 future_positions_gt=gt_future_denorm.cpu(),
                                 future_positions_pred=pred_future_denorm.cpu(),
                                 past_mask=hist_mask.cpu(),
                                 future_mask_gt=future_mask.cpu(), # Use the future mask
                                 title=f"Pred vs GT - {vis_title_base} (Eval)",
                                 save_path=pred_vs_gt_path,
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
        print(f" Mean L1 (MAE): {mean_l1:.4f}")
        print(f" Mean RMSE: {mean_rmse:.4f}")
        print(f" Mean FDE: {mean_fde:.4f}")
    else:
        print("Error: No batches were successfully processed.")
        mean_l1 = float('inf')
        mean_rmse = float('inf')
        mean_fde = float('inf')

    # --- Save Results ---
    results = {
        'model_path': args.model_path,
        'config': vars(config), # Save config used
        'eval_args': vars(args), # Save evaluation args
        'mean_l1': mean_l1,
        'mean_rmse': mean_rmse,
        'mean_fde': mean_fde,
        'num_test_samples_processed': total_valid_samples, # Report processed samples
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'comment': args.comment
    }

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {results_path}")

    return results


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Evaluation results will be saved to: {args.output_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and config
    try:
        model, config = load_model_and_config(args.model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run evaluation
    evaluate(model, config, args)

if __name__ == "__main__":
    main() 