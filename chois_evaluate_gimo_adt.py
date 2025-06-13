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
from model.chois_gimo_adt_model import SimpleTrajectoryTransformer
from dataset.gimo_multi_sequence_dataset import GIMOMultiSequenceDataset
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

def extract_pointcloud_along_trajectory(scene_pc: torch.Tensor, trajectory: torch.Tensor, radius: float) -> Optional[torch.Tensor]:
    """
    Extracts a point cloud from the scene that is within a certain radius of any point on the trajectory.
    """
    if scene_pc is None or trajectory is None or scene_pc.numel() == 0 or trajectory.numel() == 0:
        return None
    
    device = trajectory.device
    scene_pc = scene_pc.to(device)
    radius_sq = radius ** 2
    trajectory_expanded = trajectory.unsqueeze(1)
    scene_pc_expanded = scene_pc.unsqueeze(0)
    dist_sq = torch.sum((trajectory_expanded - scene_pc_expanded) ** 2, dim=2)
    min_dist_sq, _ = torch.min(dist_sq, dim=0)
    mask = min_dist_sq <= radius_sq
    return scene_pc[mask]

def setup_logger(output_dir):
    """Set up logger to both write to console and to file."""
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    log_file = os.path.join(output_dir, 'evaluation.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("=" * 50)
    logger.info(f"Evaluation started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Simple Transformer ADT trajectory prediction model")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pth file)")
    parser.add_argument("--output_dir", type=str, default="evaluation_transformer_adt", help="Directory to save evaluation results")
    parser.add_argument("--adt_dataroot", type=str, default=None, help="Override path to ADT dataroot")
    parser.add_argument("--test_split_file", type=str, default=None, help="Override path to test split file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--visualize", action="store_true", default=True, help="Enable visualization of trajectories")
    parser.add_argument("--num_vis_samples", type=int, default=500, help="Number of trajectory samples to visualize")
    parser.add_argument("--conditioning_strategy", type=str, default='history_fraction', choices=['history_fraction', 'chois_original'], help="Conditioning strategy for the model input.")
    parser.add_argument("--waypoint_interval", type=int, default=30, help="Interval for waypoints in 'chois_original' strategy.")
    parser.add_argument("--use_rerun", action="store_true", default=True, help="Enable Rerun visualization")
    parser.add_argument("--pointcloud_downsample_factor", type=int, default=1, help="Factor for downsampling point clouds in Rerun visualization")
    parser.add_argument("--rerun_line_width", type=float, default=0.02, help="Width of lines in Rerun visualization")
    parser.add_argument("--rerun_point_size", type=float, default=0.03, help="Size of points in Rerun visualization")
    parser.add_argument("--rerun_show_arrows", action="store_true", default=True, help="Show orientation arrows in Rerun visualization")
    parser.add_argument("--rerun_arrow_length", type=float, default=0.2, help="Length of orientation arrows in Rerun visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--global_cache_dir", type=str, default=None, help="Path to a shared global directory for trajectory cache")
    parser.add_argument('--use_trajectory_pointcloud', type=float, default=1.0, metavar='RADIUS', help='Extract and use a trajectory-specific point cloud for visualization with the specified radius.')
    parser.add_argument("--show_ori_arrows", action="store_true", help="Show orientation arrows in visualizations")
    parser.add_argument("--num_top_worst_to_save", type=int, default=10, help="Number of best and worst Rerun recordings and visualizations to save.")
    parser.add_argument("--method_name", type=str, default='baseline', help="Method name for visualization (e.g., 'baseline', 'our'). Overrides checkpoint config if provided.")
    parser.add_argument("--pred_color", type=str, default='orange', help="Prediction color for visualization (e.g., 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow'). Overrides checkpoint config if provided.")

    return parser.parse_args()

def load_model_and_config(checkpoint_path, device, logger, args=None):
    """
    Load SimpleTrajectoryTransformer and its config from a checkpoint.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        error_msg = f"Checkpoint file not found: {checkpoint_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'config'.")
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")
        
    config_dict = checkpoint['config']
    config = argparse.Namespace(**config_dict)
    
    # Add missing visualization parameters with defaults if they don't exist
    if not hasattr(config, 'method_name'):
        config.method_name = 'baseline'
        logger.info("Added missing method_name parameter with default value: baseline")
    
    if not hasattr(config, 'pred_color'):
        config.pred_color = 'red'
        logger.info("Added missing pred_color parameter with default value: red")
    
    # Override with command line arguments if provided
    if args:
        if args.method_name is not None:
            config.method_name = args.method_name
            logger.info(f"Overriding method_name with command line value: {args.method_name}")
        
        if args.pred_color is not None:
            config.pred_color = args.pred_color
            logger.info(f"Overriding pred_color with command line value: {args.pred_color}")
    
    logger.info("Configuration loaded from checkpoint:")
    logger.info(str(config))

    logger.info("Using SimpleTrajectoryTransformer.")
    model = SimpleTrajectoryTransformer(config).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    param_count_millions = param_count / 1_000_000
    logger.info(f"Loaded model with {param_count_millions:.2f}M parameters.")
    
    best_epoch = checkpoint.get('epoch', 0)
    if best_epoch > 0:
        logger.info(f"Checkpoint was saved at epoch: {best_epoch}")
    else:
        logger.warning("Warning: Epoch number not found in checkpoint.")
        
    return model, config, best_epoch

def denormalize_trajectory(normalized_trajectory, normalization_params):
    """Denormalize a trajectory using provided parameters."""
    if not normalization_params['is_normalized']:
        return normalized_trajectory

    scene_min_val = normalization_params.get('scene_min')
    scene_max_val = normalization_params.get('scene_max')
    scene_scale_val = normalization_params.get('scene_scale')

    if not all(isinstance(p, torch.Tensor) for p in [scene_min_val, scene_max_val, scene_scale_val]):
         print("Warning: Invalid normalization parameters found. Returning original trajectory.")
         return normalized_trajectory

    scene_min = scene_min_val.to(normalized_trajectory.device)
    scene_scale = scene_scale_val.to(normalized_trajectory.device)

    denormalized = (normalized_trajectory + 1.0) / 2.0
    denormalized = denormalized * scene_scale + scene_min
    return denormalized

def evaluate(model, config, args, best_epoch, logger):
    """
    Run the evaluation loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info("Setting up test dataset...")
    checkpoint_dir = os.path.dirname(args.model_path)
    val_split_path = os.path.join(checkpoint_dir, 'val_sequences.txt')
    
    test_sequences = []
    if os.path.exists(val_split_path):
            with open(val_split_path, 'r') as f:
                test_sequences = [line.strip() for line in f if line.strip()]
                 
            if not test_sequences:
                logger.error("No test sequences found. Please check val_sequences.txt in the model directory.")
                return None

    if args.global_cache_dir:
        eval_cache_dir = args.global_cache_dir
    else:
        eval_cache_dir = os.path.join(args.output_dir, 'trajectory_cache_eval')
        os.makedirs(eval_cache_dir, exist_ok=True)

    test_dataset = GIMOMultiSequenceDataset(
        sequence_paths=test_sequences,
        config=config,
        cache_dir=eval_cache_dir,
        use_cache=True
    )

    if len(test_dataset) == 0:
        logger.error("Test dataset is empty.")
        return None

    eval_collate_func = partial(gimo_collate_fn, dataset=test_dataset, num_sample_points=config.sample_points)
    eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, collate_fn=eval_collate_func)
    
    logger.info(f"Test DataLoader ready with {len(test_dataset)} samples.")

    total_l1, total_rmse_ade, total_fde, total_frechet, total_angular_cosine = 0.0, 0.0, 0.0, 0.0, 0.0
    total_valid_samples = 0
    all_samples_metrics = []
    
    vis_output_dir = os.path.join(args.output_dir, "visualizations")
    if args.visualize:
        os.makedirs(vis_output_dir, exist_ok=True)
        logger.info(f"Visualizations will be saved to {vis_output_dir}")

    per_sample_rrd_basedir = None
    if args.use_rerun and HAS_RERUN:
        per_sample_rrd_basedir = os.path.join(args.output_dir, "rerun_visualizations_per_sample")
        os.makedirs(per_sample_rrd_basedir, exist_ok=True)
        logger.info(f"Rerun per-sample .rrd files will be saved to: {per_sample_rrd_basedir}")
        
        # Test basic rerun functionality
        test_rrd_path = os.path.join(per_sample_rrd_basedir, "test_rerun.rrd")
        try:
            logger.info("Testing basic Rerun functionality...")
            rr.init("test_recording", spawn=False)
            rr.set_time_sequence("frame", 0)
            
            # Log a simple test point
            test_point = np.array([[0.0, 0.0, 0.0]])
            rr.log("test/point", rr.Points3D(positions=test_point, colors=[1, 0, 0, 1], radii=0.1))
            
            # Save test file
            rr.save(test_rrd_path)
            
            # Check if test file was created
            if os.path.exists(test_rrd_path):
                test_file_size = os.path.getsize(test_rrd_path)
                logger.info(f"Rerun test successful - created file of size {test_file_size} bytes")
                # Clean up test file
                os.remove(test_rrd_path)
            else:
                logger.error("Rerun test failed - no file created")
                
        except Exception as e:
            logger.error(f"Rerun test failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    with torch.no_grad():
        progress_bar = tqdm(eval_loader, desc="Evaluating")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                if 'full_poses' in batch and 'poses' not in batch:
                    batch['poses'] = batch.pop('full_poses')
                if 'full_attention_mask' in batch and 'attention_mask' not in batch:
                    batch['attention_mask'] = batch.pop('full_attention_mask')

                batch['poses'] = batch['poses'].float().to(device)
                batch['attention_mask'] = batch['attention_mask'].to(device)
                
                predicted_full_trajectory, cond_mask = model(batch, history_fraction=config.history_fraction, conditioning_strategy=args.conditioning_strategy, waypoint_interval=args.waypoint_interval)

                for i in range(batch['poses'].shape[0]):
                    gt_full_poses = batch['poses'][i]
                    pred_full_poses = predicted_full_trajectory[i]
                    mask_full = batch['attention_mask'][i]
                    
                    actual_length = torch.sum(mask_full).int().item()
                    if actual_length < 2: continue

                    history_length = max(1, int(np.floor(actual_length * config.history_fraction)))

                    # Get positional data for metrics
                    gt_future_pos = gt_full_poses[history_length:actual_length, :3]
                    pred_future_pos = pred_full_poses[history_length:actual_length, :3]
                    future_mask = mask_full[history_length:actual_length]

                    if future_mask.sum() == 0: continue

                    l1, rmse, fde, frechet, angular_cos = compute_metrics_for_sample(pred_future_pos, gt_future_pos, future_mask)

                    total_l1 += l1.item()
                    total_rmse_ade += rmse.item()
                    total_fde += fde.item()
                    total_frechet += frechet.item()
                    total_angular_cosine += angular_cos.item()
                    total_valid_samples += 1

                    # --- Visualization Logic (Restored and Adapted) ---
                    if args.visualize or (args.use_rerun and HAS_RERUN):
                        obj_name = batch['object_name'][i] if 'object_name' in batch else f"unknown_{i}"
                        seg_idx = batch['segment_idx'][i].item() if 'segment_idx' in batch and batch['segment_idx'][i].item() != -1 else None
                        current_sequence_path = batch['sequence_path'][i]
                        sequence_name_extracted = os.path.splitext(os.path.basename(current_sequence_path))[0]

                        filename_base = f"{obj_name}_seq_{sequence_name_extracted}_seg{seg_idx if seg_idx is not None else 'NA'}_batch{batch_idx}"
                        vis_title_base = f"{obj_name} (Seq: {sequence_name_extracted}, Seg: {seg_idx if seg_idx is not None else 'NA'})"

                        # Prepare full data for visualization
                        gt_hist_pos = gt_full_poses[:history_length, :3]
                        gt_hist_rot = gt_full_poses[:history_length, 3:]
                        gt_future_rot = gt_full_poses[history_length:actual_length, 3:]
                        pred_future_rot = pred_full_poses[history_length:actual_length, 3:]
                        hist_mask = mask_full[:history_length]

                        # Matplotlib visualization
                        if args.visualize:
                             pred_vs_gt_path = os.path.join(vis_output_dir, f"{filename_base}_prediction_vs_gt.png")
                             visualize_prediction(
                                past_positions=transform_coords_for_visualization(gt_hist_pos.cpu()),
                                future_positions_gt=transform_coords_for_visualization(gt_future_pos.cpu()),
                                future_positions_pred=transform_coords_for_visualization(pred_future_pos.cpu()),
                                 past_mask=hist_mask.cpu(),
                                future_mask_gt=future_mask.cpu(),
                                title=f"Pred vs GT - {vis_title_base}",
                                 save_path=pred_vs_gt_path,
                                 segment_idx=seg_idx,
                                 show_orientation=args.show_ori_arrows,
                                past_orientations=gt_hist_rot.cpu(),
                                future_orientations_gt=gt_future_rot.cpu(),
                                future_orientations_pred=pred_future_rot.cpu()
                             )
                             
                        # Rerun visualization
                        if args.use_rerun and HAS_RERUN and per_sample_rrd_basedir:
                            try:
                                sample_recording_name = filename_base
                                rrd_file_path = os.path.join(per_sample_rrd_basedir, f"{sample_recording_name}.rrd")
                                
                                logger.info(f"Creating Rerun visualization for: {sample_recording_name}")
                                
                                # Initialize rerun recording
                                sample_recording_name = filename_base # e.g., "obj_seq_seg_batch"
                                logger.info(f"Attempting Rerun visualization for sample: {sample_recording_name}")

                                # Initialize Rerun for THIS SPECIFIC SAMPLE.
                                sample_rerun_initialized, returned_sample_rrd_path = initialize_rerun(
                                    recording_name=sample_recording_name, 
                                    spawn=False, # No viewer per sample
                                    output_dir=per_sample_rrd_basedir 
                                )
                                
                                # Set a base time to ensure data is logged
                                rr.set_time_sequence("animation_step", 0)
                                
                                # Process point cloud
                                point_cloud = batch['point_cloud'][i].cpu()
                                if args.pointcloud_downsample_factor > 1:
                                    point_cloud = downsample_point_cloud(point_cloud, args.pointcloud_downsample_factor)
                                
                                # Log some debug info
                                logger.info(f"  - Past trajectory: {gt_hist_pos.shape[0]} points")
                                logger.info(f"  - Future GT: {gt_future_pos.shape[0]} points") 
                                logger.info(f"  - Future Pred: {pred_future_pos.shape[0]} points")
                                logger.info(f"  - Point cloud: {point_cloud.shape[0] if point_cloud is not None else 0} points")
                                
                                # Visualize trajectory in rerun (this actually records the data)
                                success = visualize_trajectory_rerun(
                                    past_positions=transform_coords_for_visualization(gt_hist_pos.cpu()),
                                    future_positions_gt=transform_coords_for_visualization(gt_future_pos.cpu()),
                                    future_positions_pred=transform_coords_for_visualization(pred_future_pos.cpu()),
                                    past_mask=hist_mask,
                                    future_mask_gt=future_mask,
                                    point_cloud=transform_coords_for_visualization(point_cloud) if point_cloud is not None else None,
                                    past_orientations=gt_hist_rot.cpu(),
                                    future_orientations_gt=gt_future_rot.cpu(),
                                    future_orientations_pred=pred_future_rot.cpu(),
                                    object_name=obj_name, 
                                    sequence_name=sequence_name_extracted, 
                                    segment_idx=seg_idx, 
                                    arrow_length=args.rerun_arrow_length,
                                    line_width=args.rerun_line_width,
                                    point_size=args.rerun_point_size,
                                    show_arrows=args.rerun_show_arrows,
                                    method_name=config.method_name,
                                    pred_color=config.pred_color
                                )
                                
                                if success:
                                    logger.info(f"  - Visualization successful, saving to: {rrd_file_path}")
                                    # Save the recording after data has been logged
                                    rr.save(rrd_file_path)
                                    
                                    # Verify file was created and has content
                                    if os.path.exists(rrd_file_path):
                                        file_size = os.path.getsize(rrd_file_path)
                                        logger.info(f"  - File saved successfully, size: {file_size} bytes")
                                        if file_size < 1000:  # Less than 1KB suggests empty file
                                            logger.warning(f"  - Warning: File size is very small ({file_size} bytes), may be empty")
                                    else:
                                        logger.error(f"  - Error: File was not created at {rrd_file_path}")
                                else:
                                    logger.error(f"  - Visualization failed for {sample_recording_name}")
                                    
                            except Exception as e:
                                logger.error(f"Error creating Rerun visualization for {sample_recording_name}: {e}")
                                import traceback
                                logger.error(traceback.format_exc())

            except Exception as e:
                 logger.error(f"Error processing batch {batch_idx}: {e}")
                 import traceback
                 logger.error(traceback.format_exc())
                 continue

    if total_valid_samples > 0:
        mean_l1 = total_l1 / total_valid_samples
        mean_rmse = total_rmse_ade / total_valid_samples
        mean_fde = total_fde / total_valid_samples
        mean_frechet = total_frechet / total_valid_samples
        mean_angular_cosine = total_angular_cosine / total_valid_samples

        logger.info(f"--- Evaluation Complete ---")
        logger.info(f" Model from Epoch: {best_epoch}")
        logger.info(f" Mean L1 (MAE): {mean_l1:.4f}")
        logger.info(f" Mean RMSE: {mean_rmse:.4f}")
        logger.info(f" Mean FDE: {mean_fde:.4f}")
        logger.info(f" Mean Fréchet: {mean_frechet:.4f}")
        logger.info(f" Mean Angular_Cos: {mean_angular_cosine:.4f}")
        
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
    }
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Evaluation results saved to {results_path}")
    

def main():
    """Main evaluation function."""
    args = parse_args()
    
    output_dir = args.output_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.dirname(os.path.normpath(args.model_path))
    comment = os.path.basename(model_dir)
    output_dir = f"{output_dir}/{timestamp}_{comment}"
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(output_dir)
    logger.info(f"Evaluating model: {args.model_path}")
    logger.info(f"Results will be saved to: {output_dir}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Command line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    try:
        model, config, best_epoch = load_model_and_config(args.model_path, device, logger, args)
        evaluate(model, config, args, best_epoch, logger)
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

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
#
# 7. Evaluation with Rerun visualization including arrows and semantic bounding boxes with categories
# python evaluate_gimo_adt.py --model_path ./checkpoints/best_model.pth --use_rerun --rerun_show_arrows --rerun_show_semantic_bboxes --num_vis_samples 10
#
# 8. Customized Rerun visualization settings with semantic object categories
# python evaluate_gimo_adt.py --model_path ./checkpoints/best_model.pth --use_rerun --rerun_arrow_length 0.3 --rerun_line_width 0.03 --pointcloud_downsample_factor 5
#
# 9. Evaluation with both Matplotlib and Rerun visualization showing semantic objects
# python evaluate_gimo_adt.py --model_path ./checkpoints/best_model.pth --visualize --visualize_bbox --use_rerun --rerun_show_arrows --rerun_show_semantic_bboxes 
#
# Note: Each semantic bounding box will now appear as an individual object in Rerun,
#       named by its category (e.g., "food_object_0", "dining_table_1", "cutting_board_2") with category-specific colors
#       Category names are cleaned for Rerun paths (spaces become underscores, special characters removed) 