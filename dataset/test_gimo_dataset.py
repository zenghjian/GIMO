#!/usr/bin/env python3
"""
Test script for GIMOMultiSequenceDataset to verify loading and data access.
Includes testing of GIMO_ADT_Model structure and visualization.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm # Import colormap library
import argparse 
from torch.utils.data.dataloader import default_collate # Import default_collate
from functools import partial # Import partial

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add parent directory to path to access utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our dataset and NEW config/model
from config.adt_config import ADTObjectMotionConfig
from dataset.gimo_multi_sequence_dataset import GIMOMultiSequenceDataset
from model.gimo_adt_model import GIMO_ADT_Model
from utils.visualization import visualize_trajectory, visualize_prediction, visualize_full_trajectory, visualize_pointcloud # Import from utils

# --- Custom Collate Function (Copied from train_adt.py) ---
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
    pc_dict = {}
    batch_dataset_indices = [item['dataset_idx'] for item in batch]
    
    for i, dataset_idx in enumerate(batch_dataset_indices):
        if dataset_idx not in pc_dict:
            point_cloud = dataset.get_scene_pointcloud(dataset_idx)
            if point_cloud is None:
                print(f"Warning: Failed to get point cloud for dataset_idx {dataset_idx} in batch. Using zeros.")
                point_cloud = torch.zeros((num_sample_points, 3), dtype=torch.float32)
            elif isinstance(point_cloud, np.ndarray):
                point_cloud = torch.from_numpy(point_cloud).float()
            
            if point_cloud.shape[0] >= num_sample_points:
                indices = np.random.choice(point_cloud.shape[0], num_sample_points, replace=False)
                sampled_pc = point_cloud[indices]
            else:
                print(f"Warning: Point cloud for dataset_idx {dataset_idx} has only {point_cloud.shape[0]} points. Sampling with replacement to get {num_sample_points}.")
                indices = np.random.choice(point_cloud.shape[0], num_sample_points, replace=True)
                sampled_pc = point_cloud[indices]
                
            pc_dict[dataset_idx] = sampled_pc
            
    point_cloud_batch_list = [pc_dict[idx] for idx in batch_dataset_indices]
    batched_point_clouds = torch.stack(point_cloud_batch_list, dim=0)

    batch_copy = [{k: v for k, v in item.items() if k != 'dataset_idx'} for item in batch]
    collated_batch = default_collate(batch_copy)
    
    collated_batch['point_cloud'] = batched_point_clouds
    
    return collated_batch
# --------------------------------------------------------

def main():
    # --- Configuration and Argument Parsing ---
    print("Parsing configuration and arguments...")
    # Instantiate the main config parser
    adt_config_parser = ADTObjectMotionConfig()
    # Add test-specific arguments directly to it
    adt_config_parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to model checkpoint (.pth) to load for testing.')
    adt_config_parser.add_argument('--max_vis_trajectories', type=int, default=10, help='Maximum number of trajectories to visualize')
    # Parse all arguments together
    config = adt_config_parser.parse_args()
    
    print("\nConfiguration loaded:")
    print(config) 
    if config.load_checkpoint:
        print(f"\nCheckpoint to load specified: {config.load_checkpoint}")
    # -----------------------------------------
    
    # --- Device Setup --- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    # ----------------------

    # --- Dataset and DataLoader ---
    print("Loading dataset...")
    # Set the path to our test sequence (can be overridden by config default)
    # sequence_path = config.adt_dataroot # Old way - points to parent dir
    # Explicitly point to one sequence directory for testing
    sequence_path = os.path.join(config.adt_dataroot, "Apartment_release_clean_seq131_M1292") 
    print(f"Explicitly using test sequence: {sequence_path}")
    
    # Create output directory for visualizations
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup cache directory
    cache_dir = os.path.join(output_dir, 'trajectory_cache')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    
    print(f"Testing GIMOMultiSequenceDataset with sequence: {sequence_path}")
    
    # Create dataset with settings from config
    dataset = GIMOMultiSequenceDataset(
        sequence_paths=[sequence_path],
        config=config,  # Pass the config object 
        cache_dir=cache_dir  # Still explicitly set cache_dir
    )
    
    print(f"Dataset loaded with {len(dataset)} trajectories")
    
    # Check if we have any data
    if len(dataset) == 0:
        print("Error: No trajectories found in dataset!")
        return
    
    print(f"Dataset size: {len(dataset)}")

    # --- Model Initialization ---
    print("\nInstantiating GIMO_ADT_Model...")
    model = GIMO_ADT_Model(config).to(device)
    
    # --- Load Weights if Checkpoint Specified ---
    if config.load_checkpoint and os.path.exists(config.load_checkpoint):
        print(f"Loading weights from checkpoint: {config.load_checkpoint}")
        try:
            checkpoint = torch.load(config.load_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Successfully loaded model state_dict from checkpoint.")
            else:
                model.load_state_dict(checkpoint)
                print("Successfully loaded state_dict directly from checkpoint file.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Using randomly initialized model.")
    elif config.load_checkpoint:
        print(f"Warning: Checkpoint path specified but not found: {config.load_checkpoint}. Using randomly initialized model.")
    else:
        print("No checkpoint specified. Using randomly initialized model.")
    # -----------------------------------------

    print(f"Model initialized. Parameter count: {sum(p.numel() for p in model.parameters())}")
    model.eval() # Set model to evaluation mode

    # Try to get point cloud
    print("\nAttempting to get scene point cloud...")
    pointcloud = dataset.get_scene_pointcloud()
    if pointcloud is not None:
        print(f"Point cloud loaded successfully! Shape: {pointcloud.shape}")
        
        # --- Add detailed point cloud checks ---
        print(f"  Point cloud data type: {type(pointcloud)}")
        if isinstance(pointcloud, np.ndarray):
             # Check dimensions
            if pointcloud.ndim == 2 and pointcloud.shape[1] == 3:
                print("  Shape check passed (N, 3).")
                # Calculate and print min/max for each coordinate
                min_coords = np.min(pointcloud, axis=0)
                max_coords = np.max(pointcloud, axis=0)
                print(f"  Min Coordinates (X, Y, Z): ({min_coords[0]:.3f}, {min_coords[1]:.3f}, {min_coords[2]:.3f})")
                print(f"  Max Coordinates (X, Y, Z): ({max_coords[0]:.3f}, {max_coords[1]:.3f}, {max_coords[2]:.3f})")
            else:
                print(f"  Warning: Unexpected point cloud shape {pointcloud.shape}. Expected (N, 3).")
        elif isinstance(pointcloud, torch.Tensor):
            print("  Warning: Point cloud is a Tensor, expected NumPy array from this dataset method.")
            if pointcloud.ndim == 2 and pointcloud.shape[1] == 3:
                 print("  Shape check passed (N, 3).")
            else:
                 print(f"  Warning: Unexpected point cloud shape {pointcloud.shape}. Expected (N, 3).")
        # ---------------------------------------
        
        visualize_pointcloud(
            pointcloud,
            title="Scene Point Cloud",
            save_path=os.path.join(output_dir, "scene_pointcloud.png")
        )
        print(f"Point cloud shape: {pointcloud.shape}")
    else:
        # Updated message
        print("Point cloud data was not loaded or is unavailable for this sequence.")
    
    # ----- NEW SECTION: Visualize multiple trajectories with segment information -----
    print("\n--- Visualizing Multiple Trajectories with Segment Information ---")
    
    # Define maximum number of trajectories to visualize (to avoid too many plots)
    max_vis_trajectories = config.max_vis_trajectories  # Use the parameter from config
    
    # Create a separate directory for multiple trajectory visualizations
    multi_traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(multi_traj_dir, exist_ok=True)
    
    # Set up for inference with model
    model.eval()
    
    # Determine how many trajectories to visualize (limited by max_vis_trajectories)
    num_to_visualize = min(len(dataset), max_vis_trajectories)
    print(f"Visualizing {num_to_visualize} out of {len(dataset)} trajectories")
    
    # Process multiple trajectories for visualization
    for traj_idx in range(num_to_visualize):
        # Get the trajectory from the dataset
        traj_item = dataset[traj_idx]
        object_name = traj_item['object_name']
        object_id = traj_item['object_id'].item() if isinstance(traj_item['object_id'], torch.Tensor) else traj_item['object_id']
        
        # Get segment information if available
        # Correctly access segment_idx from the sample
        segment_idx = traj_item['segment_idx'].item() if 'segment_idx' in traj_item and traj_item['segment_idx'].item() != -1 else None

        # Create a descriptive filename base for this trajectory
        if segment_idx is not None:
            filename_base = f"{object_name}_id{object_id}_seg{segment_idx}"
            vis_title_base = f"{object_name} (ID: {object_id}, Segment: {segment_idx})"
        else:
            filename_base = f"{object_name}_id{object_id}"
            vis_title_base = f"{object_name} (ID: {object_id})"
        
        # Get trajectory data 
        full_positions = traj_item['full_positions']
        full_mask = traj_item.get('full_attention_mask')
        
        # Process through model (need to add batch dimension)
        with torch.no_grad():
            full_positions_batch = full_positions.unsqueeze(0).to(device)
            # Also get point cloud for this trajectory
            dataset_idx = traj_item.get('dataset_idx', 0)
            point_cloud = dataset.get_scene_pointcloud(dataset_idx)
            if isinstance(point_cloud, np.ndarray):
                point_cloud = torch.from_numpy(point_cloud).float()
            
            # Sample points if needed
            if point_cloud.shape[0] > config.sample_points:
                indices = np.random.choice(point_cloud.shape[0], config.sample_points, replace=False)
                point_cloud = point_cloud[indices]
            
            # Add batch dimension and send to device
            point_cloud_batch = point_cloud.unsqueeze(0).to(device)
            
            # Run model inference
            predicted_trajectory = model(full_positions_batch, point_cloud_batch)
            
            # Remove batch dimension
            predicted_trajectory = predicted_trajectory[0]
        
        # --- Dynamic Split Calculation for Visualization ---
        if full_mask is None:
            print(f"Warning: Cannot perform dynamic split for visualization for {object_name} (idx {traj_idx}) - mask missing.")
            actual_length = full_positions.shape[0]
            dynamic_history_length = int(np.floor(actual_length * config.history_fraction))
        else:
            actual_length = torch.sum(full_mask).int().item()
            dynamic_history_length = int(np.floor(actual_length * config.history_fraction))
            dynamic_history_length = max(1, min(dynamic_history_length, actual_length)) # Ensure valid length

        # Split the GT trajectories for visualization
        past_positions = full_positions[:dynamic_history_length]
        future_positions_gt = full_positions[dynamic_history_length:actual_length] # Slice up to actual length
        
        # Split the GT mask if available
        past_mask = full_mask[:dynamic_history_length] if full_mask is not None else None
        future_mask = full_mask[dynamic_history_length:actual_length] if full_mask is not None else None
        
        # Slice the prediction dynamically
        pred_past_vis = predicted_trajectory[:dynamic_history_length] 
        future_positions_pred = predicted_trajectory[dynamic_history_length:actual_length] # Slice up to actual length
        # ----------------------------------------------------
        
        # Visualize trajectory split (history/future GT)
        trajectory_split_path = os.path.join(multi_traj_dir, f"{filename_base}_trajectory_split.png")
        visualize_trajectory(
            past_positions=past_positions,
            future_positions=future_positions_gt,
            past_mask=past_mask,
            future_mask=future_mask,
            title=f"Trajectory Split - {vis_title_base}",
            save_path=trajectory_split_path,
            segment_idx=segment_idx
        )
        
        # Visualize prediction vs. ground truth
        prediction_path = os.path.join(multi_traj_dir, f"{filename_base}_prediction_vs_gt.png")
        visualize_prediction(
            past_positions=past_positions,
            future_positions_gt=future_positions_gt,
            future_positions_pred=future_positions_pred,
            past_mask=past_mask,
            future_mask_gt=future_mask,
            title=f"Prediction vs. GT - {vis_title_base}",
            save_path=prediction_path,
            segment_idx=segment_idx
        )
        
        # Visualize the full trajectory (complete motion path)
        full_traj_path = os.path.join(multi_traj_dir, f"{filename_base}_full_trajectory.png")
        visualize_full_trajectory(
            positions=full_positions,
            attention_mask=full_mask,
            title=f"Full Trajectory - {vis_title_base}",
            save_path=full_traj_path,
            segment_idx=segment_idx
        )
        
        print(f"Visualized trajectory {traj_idx+1}/{num_to_visualize}: {object_name}")
    
    print(f"\nAll visualizations saved to {multi_traj_dir}")
    # ------------------------------------------------------------------------
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 