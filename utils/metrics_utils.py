#!/usr/bin/env python3
"""
Metrics and data processing utilities for GIMO ADT training and evaluation.
"""

import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from functools import partial


def transform_coords_for_visualization(tensor_3d: torch.Tensor) -> torch.Tensor:
    """
    Applies (x, y, z) -> (x, -z, y) transformation to a 3D tensor for visualization.
    
    Args:
        tensor_3d: Input tensor with last dimension of 3 (x, y, z coordinates)
        
    Returns:
        torch.Tensor: Transformed tensor with (x, -z, y) coordinates
    """
    if tensor_3d is None or tensor_3d.numel() == 0:
        return tensor_3d
    
    # Ensure it's a tensor
    if not isinstance(tensor_3d, torch.Tensor):
        tensor_3d = torch.tensor(tensor_3d)

    if tensor_3d.shape[-1] != 3:
        # print(f"Warning: transform_coords_for_visualization expects last dim to be 3, got {tensor_3d.shape}. Skipping transformation.")
        return tensor_3d

    x = tensor_3d[..., 0]
    y = tensor_3d[..., 1]
    z = tensor_3d[..., 2]
    
    transformed_tensor = torch.stack((x, -z, y), dim=-1)
    return transformed_tensor


def compute_metrics_for_sample(pred_future, gt_future, future_mask):
    """
    Compute L1 Mean, L2 Mean (for RMSE), and FDE for a single trajectory sample.

    Args:
        pred_future (torch.Tensor): Predicted future trajectory [Fut_Len, 3] (position only)
        gt_future (torch.Tensor): Ground truth future trajectory [Fut_Len, 3] (position only)
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
    l1_diff = torch.abs(pred_future - gt_future)  # Shape: [Fut_Len, 3]
    # Calculate per-timestep L2 distance (Euclidean norm)
    l2_dist_per_step = torch.norm(pred_future - gt_future, dim=-1)  # Shape: [Fut_Len]

    # Expand mask to match 3D coordinates (for L1)
    future_mask_expanded = future_mask.unsqueeze(-1).expand_as(l1_diff)  # Shape: [Fut_Len, 3]

    # Count valid points (use original 1D mask)
    num_valid_points = future_mask.sum()
    if num_valid_points == 0:
        # Return zeros if no valid points to avoid division by zero
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
    num_valid_coords = num_valid_points * 3  # Total number of valid coordinate values

    # --- L1 Mean (MAE) Calculation ---
    # Mask out invalid points
    masked_l1_diff = l1_diff * future_mask_expanded
    # Sum distances over all valid coordinates
    sum_l1_diff = masked_l1_diff.sum()
    # Calculate average L1 distance
    l1_mean = sum_l1_diff / num_valid_coords

    # --- RMSE / ADE (L2) Calculation ---
    # Mask out invalid points
    masked_l2_dist = l2_dist_per_step * future_mask  # Use 1D mask here
    # Sum L2 distances over all valid timesteps
    sum_l2_dist = masked_l2_dist.sum()
    # Calculate average L2 distance over valid timesteps
    rmse_ade = sum_l2_dist / num_valid_points

    # --- FDE Calculation ---
    # Find the index of the last valid point (using 1D mask)
    last_valid_index = num_valid_points.long() - 1
    last_valid_index = torch.clamp(last_valid_index, min=0)  # Clamp index to avoid negative indices

    # Get the predicted and ground truth points at the final valid timestep
    final_pred_point = pred_future[last_valid_index]
    final_gt_point = gt_future[last_valid_index]

    # Calculate FDE as the L2 distance between these final points
    fde = torch.norm(final_pred_point - final_gt_point, dim=-1)

    return l1_mean, rmse_ade, fde


def gimo_collate_fn(batch, dataset, num_sample_points):
    """
    Custom collate function to handle trajectory-specific point clouds.
    
    Args:
        batch (list): A list of sample dictionaries from GIMOMultiSequenceDataset.
                      Each dict must contain 'dataset_idx' and other required data.
        dataset (GIMOMultiSequenceDataset): The instance of the dataset being used.
                                            Needed to access get_scene_pointcloud.
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
                
            if i == 0:  # Only print for first item to avoid log spam
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
        
        if i == 0:  # Only print for first item to avoid log spam
            print(f"Using full scene point cloud with {point_cloud.shape[0]} points")
    
        # Sample the point cloud to ensure consistent size
        if point_cloud.shape[0] >= num_sample_points:
            # Randomly sample points without replacement
            indices = np.random.choice(point_cloud.shape[0], num_sample_points, replace=False)
            sampled_pc = point_cloud[indices]
        else:
            # If not enough points, sample with replacement
            if i == 0:  # Only print for first item to avoid log spam
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
        # bbox_corners should be handled by default_collate, so no need to exclude it here.
        batch_copy.append(item_copy)
    
    # Collate the rest using default_collate
    collated_batch = default_collate(batch_copy)
    
    # Add the batched point clouds
    collated_batch['point_cloud'] = batched_point_clouds
    
    return collated_batch
