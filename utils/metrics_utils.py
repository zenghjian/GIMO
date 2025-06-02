#!/usr/bin/env python3
"""
Metrics and data processing utilities for GIMO ADT training and evaluation.
"""

import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from tqdm import tqdm

# Import geometry utilities for rotation conversion
try:
    from utils.geometry_utils import convert_rotation_to_euler
    HAS_GEOMETRY_UTILS = True
except ImportError:
    print("Warning: geometry_utils not found. Will use fallback rotation conversion.")
    HAS_GEOMETRY_UTILS = False


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


def convert_6d_to_euler_fallback(r6d_rotations):
    """
    Fallback function to convert 6D rotation to Euler angles when geometry_utils is not available.
    
    Args:
        r6d_rotations: [N, 6] tensor of 6D rotation representations
        
    Returns:
        ndarray: [N, 3] Euler angles (roll, pitch, yaw) in radians
    """
    if isinstance(r6d_rotations, torch.Tensor):
        r6d_rotations = r6d_rotations.detach().cpu().numpy()
    
    euler_angles = []
    for i in range(r6d_rotations.shape[0]):
        r6d = r6d_rotations[i]
        
        # Extract first two columns
        x = r6d[:3]
        y = r6d[3:]
        
        # Normalize and orthogonalize using Gram-Schmidt
        x = x / (np.linalg.norm(x) + 1e-8)
        y = y - np.dot(x, y) * x
        y = y / (np.linalg.norm(y) + 1e-8)
        z = np.cross(x, y)
        
        # Create rotation matrix
        R_matrix = np.column_stack([x, y, z])
        
        # Convert to Euler angles (ZYX convention)
        sy = np.sqrt(R_matrix[0, 0] * R_matrix[0, 0] + R_matrix[1, 0] * R_matrix[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R_matrix[2, 1], R_matrix[2, 2])
            pitch = np.arctan2(-R_matrix[2, 0], sy)
            yaw = np.arctan2(R_matrix[1, 0], R_matrix[0, 0])
        else:
            roll = np.arctan2(-R_matrix[1, 2], R_matrix[1, 1])
            pitch = np.arctan2(-R_matrix[2, 0], sy)
            yaw = 0
        
        euler_angles.append([roll, pitch, yaw])
    
    return np.array(euler_angles)


def extract_orientations_from_loader(dataloader, max_samples=None, position_dim=3):
    """
    Extract all orientation data from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader containing trajectory data
        max_samples: Maximum number of samples to process (None for all)
        position_dim: Number of position dimensions (default: 3)
        
    Returns:
        numpy.ndarray: Array of Euler angles [N, 3] in radians
    """
    all_orientations = []
    sample_count = 0
    
    print(f"Extracting orientations from dataloader with {len(dataloader)} batches...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting orientations")):
        try:
            # Get full poses from batch
            full_poses = batch['full_poses']  # [B, seq_len, 9]
            attention_mask = batch.get('full_attention_mask', None)  # [B, seq_len]
            
            batch_size, seq_len, total_dim = full_poses.shape
            rotation_dim = total_dim - position_dim
            
            # Extract rotation components (6D representation)
            rotations = full_poses[:, :, position_dim:position_dim + rotation_dim]  # [B, seq_len, 6]
            
            # Process each sample in the batch
            for i in range(batch_size):
                sample_rotations = rotations[i]  # [seq_len, 6]
                
                # Apply attention mask if available
                if attention_mask is not None:
                    mask = attention_mask[i]  # [seq_len]
                    valid_indices = torch.where(mask > 0)[0]
                    if len(valid_indices) > 0:
                        sample_rotations = sample_rotations[valid_indices]  # [valid_len, 6]
                
                # Convert 6D rotations to Euler angles
                if sample_rotations.shape[0] > 0:
                    if HAS_GEOMETRY_UTILS:
                        euler_angles = convert_rotation_to_euler(sample_rotations)
                    else:
                        euler_angles = convert_6d_to_euler_fallback(sample_rotations)
                    
                    if euler_angles is not None:
                        all_orientations.append(euler_angles)
                
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    break
            
            if max_samples is not None and sample_count >= max_samples:
                break
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    if all_orientations:
        # Concatenate all orientations
        all_orientations = np.concatenate(all_orientations, axis=0)
        print(f"Extracted {all_orientations.shape[0]} orientation samples")
        return all_orientations
    else:
        print("No valid orientations found!")
        return np.array([])


def plot_orientation_distribution(train_loader, val_loader, num_bins=36, output_path=None, max_samples_per_loader=1000, wandb_run=None):
    """
    Visualize and compare orientation angle distributions between training and validation sets.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_bins: Number of bins for histogram (default: 36, i.e., 10-degree bins for 360 degrees)
        output_path: Path to save the plot (optional)
        max_samples_per_loader: Maximum samples to extract from each loader
        wandb_run: WandB run object for logging (optional)
        
    Returns:
        dict: Statistics about the distributions
    """
    print("=== Analyzing Orientation Distributions ===")
    
    # Extract orientations from both loaders
    print("Extracting training orientations...")
    train_orientations = extract_orientations_from_loader(train_loader, max_samples=max_samples_per_loader)
    
    print("Extracting validation orientations...")
    val_orientations = extract_orientations_from_loader(val_loader, max_samples=max_samples_per_loader)
    
    if len(train_orientations) == 0 or len(val_orientations) == 0:
        print("Error: No valid orientations found in one or both datasets!")
        return {}
    
    # Convert from radians to degrees for better interpretability
    train_orientations_deg = np.rad2deg(train_orientations)
    val_orientations_deg = np.rad2deg(val_orientations)
    
    # Create figure with subplots for each angle component
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Orientation Distribution Comparison: Training vs Validation', fontsize=16)
    
    angle_names = ['Roll', 'Pitch', 'Yaw']
    colors = ['blue', 'red']
    labels = ['Training', 'Validation']
    
    stats = {}
    
    # Plot distribution for each angle component
    for angle_idx in range(3):
        # Top row: Individual histograms
        ax_individual = axes[0, angle_idx]
        
        train_angles = train_orientations_deg[:, angle_idx]
        val_angles = val_orientations_deg[:, angle_idx]
        
        # Calculate appropriate bins that cover both datasets
        all_angles = np.concatenate([train_angles, val_angles])
        bins = np.linspace(-180, 180, num_bins + 1)
        
        # Plot histograms with transparency
        ax_individual.hist(train_angles, bins=bins, alpha=0.7, color=colors[0], 
                          label=f'{labels[0]} (n={len(train_angles)})', density=True)
        ax_individual.hist(val_angles, bins=bins, alpha=0.7, color=colors[1], 
                          label=f'{labels[1]} (n={len(val_angles)})', density=True)
        
        ax_individual.set_xlabel(f'{angle_names[angle_idx]} Angle (degrees)')
        ax_individual.set_ylabel('Density')
        ax_individual.set_title(f'{angle_names[angle_idx]} Distribution')
        ax_individual.legend()
        ax_individual.grid(True, alpha=0.3)
        
        # Calculate statistics
        stats[f'{angle_names[angle_idx].lower()}_train'] = {
            'mean': np.mean(train_angles),
            'std': np.std(train_angles),
            'median': np.median(train_angles),
            'min': np.min(train_angles),
            'max': np.max(train_angles)
        }
        
        stats[f'{angle_names[angle_idx].lower()}_val'] = {
            'mean': np.mean(val_angles),
            'std': np.std(val_angles),
            'median': np.median(val_angles),
            'min': np.min(val_angles),
            'max': np.max(val_angles)
        }
        
        # Bottom row: Box plots for direct comparison
        ax_box = axes[1, angle_idx]
        data_to_plot = [train_angles, val_angles]
        box_plot = ax_box.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax_box.set_ylabel(f'{angle_names[angle_idx]} Angle (degrees)')
        ax_box.set_title(f'{angle_names[angle_idx]} Box Plot Comparison')
        ax_box.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Orientation distribution plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== Distribution Statistics Summary ===")
    for angle_idx, angle_name in enumerate(['roll', 'pitch', 'yaw']):
        train_stats = stats[f'{angle_name}_train']
        val_stats = stats[f'{angle_name}_val']
        
        print(f"\n{angle_name.upper()} Angle:")
        print(f"  Training   - Mean: {train_stats['mean']:.2f}°, Std: {train_stats['std']:.2f}°")
        print(f"  Validation - Mean: {val_stats['mean']:.2f}°, Std: {val_stats['std']:.2f}°")
        print(f"  Difference - Mean: {abs(train_stats['mean'] - val_stats['mean']):.2f}°, Std: {abs(train_stats['std'] - val_stats['std']):.2f}°")
    
    return stats


class GradientTracker:
    """
    Class to track gradients during training for different model components.
    """
    
    def __init__(self):
        self.gradient_logs = defaultdict(list)
        self.iteration_count = 0
    
    def log_gradients(self, model, loss_components=None):
        """
        Log gradient magnitudes for different model components.
        
        Args:
            model: The neural network model
            loss_components: Optional dict of individual loss components
        """
        self.iteration_count += 1
        
        # Log gradients for different model components
        component_gradients = {}
        
        # Check for different model components and log their gradients
        if hasattr(model, 'outputlayer'):
            # Final output layer (predicts full trajectory including orientation)
            output_grads = []
            for param in model.outputlayer.parameters():
                if param.grad is not None:
                    output_grads.append(param.grad.detach().flatten())
            if output_grads:
                output_grad_norm = torch.norm(torch.cat(output_grads), p=2).item()
                component_gradients['output_layer'] = output_grad_norm
        
        # Motion-related components
        if hasattr(model, 'motion_linear'):
            motion_grads = []
            for param in model.motion_linear.parameters():
                if param.grad is not None:
                    motion_grads.append(param.grad.detach().flatten())
            if motion_grads:
                motion_grad_norm = torch.norm(torch.cat(motion_grads), p=2).item()
                component_gradients['motion_embedding'] = motion_grad_norm
        
        # Scene encoder components
        if hasattr(model, 'scene_encoder'):
            scene_grads = []
            for param in model.scene_encoder.parameters():
                if param.grad is not None:
                    scene_grads.append(param.grad.detach().flatten())
            if scene_grads:
                scene_grad_norm = torch.norm(torch.cat(scene_grads), p=2).item()
                component_gradients['scene_encoder'] = scene_grad_norm
        
        # Category embedding (if exists)
        if hasattr(model, 'category_embedding'):
            cat_grads = []
            for param in model.category_embedding.parameters():
                if param.grad is not None:
                    cat_grads.append(param.grad.detach().flatten())
            if cat_grads:
                cat_grad_norm = torch.norm(torch.cat(cat_grads), p=2).item()
                component_gradients['category_embedding'] = cat_grad_norm
        
        # Motion-bbox encoder
        if hasattr(model, 'motion_bbox_encoder'):
            motion_bbox_grads = []
            for param in model.motion_bbox_encoder.parameters():
                if param.grad is not None:
                    motion_bbox_grads.append(param.grad.detach().flatten())
            if motion_bbox_grads:
                motion_bbox_grad_norm = torch.norm(torch.cat(motion_bbox_grads), p=2).item()
                component_gradients['motion_bbox_encoder'] = motion_bbox_grad_norm
        
        # Output encoder
        if hasattr(model, 'output_encoder'):
            output_enc_grads = []
            for param in model.output_encoder.parameters():
                if param.grad is not None:
                    output_enc_grads.append(param.grad.detach().flatten())
            if output_enc_grads:
                output_enc_grad_norm = torch.norm(torch.cat(output_enc_grads), p=2).item()
                component_gradients['output_encoder'] = output_enc_grad_norm
        
        # Overall model gradient norm
        total_grads = []
        for param in model.parameters():
            if param.grad is not None:
                total_grads.append(param.grad.detach().flatten())
        if total_grads:
            total_grad_norm = torch.norm(torch.cat(total_grads), p=2).item()
            component_gradients['total_model'] = total_grad_norm
        
        # Store the gradients
        for component, grad_norm in component_gradients.items():
            self.gradient_logs[component].append(grad_norm)
        
        # Also log loss components if provided
        if loss_components:
            for loss_name, loss_value in loss_components.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                self.gradient_logs[f'loss_{loss_name}'].append(loss_value)
    
    def get_logs(self):
        """
        Get the current gradient logs.
        
        Returns:
            dict: Dictionary of gradient logs
        """
        return dict(self.gradient_logs)
    
    def save_logs(self, save_path):
        """
        Save gradient logs to a file.
        
        Args:
            save_path: Path to save the logs
        """
        logs_dict = self.get_logs()
        torch.save(logs_dict, save_path)
        print(f"Gradient logs saved to: {save_path}")
    
    def load_logs(self, load_path):
        """
        Load gradient logs from a file.
        
        Args:
            load_path: Path to load the logs from
        """
        if os.path.exists(load_path):
            logs_dict = torch.load(load_path)
            self.gradient_logs = defaultdict(list, logs_dict)
            self.iteration_count = len(list(logs_dict.values())[0]) if logs_dict else 0
            print(f"Gradient logs loaded from: {load_path}")
        else:
            print(f"Warning: Gradient log file not found: {load_path}")


def plot_gradient_magnitudes(gradient_logs, component_names=None, output_path=None, 
                           window_size=50, show_losses=True, wandb_run=None, step=None):
    """
    Visualize gradient magnitudes over training iterations.
    
    Args:
        gradient_logs: Dictionary containing gradient logs for different components
        component_names: List of component names to plot (None for all)
        output_path: Path to save the plot (optional)
        window_size: Window size for moving average smoothing
        show_losses: Whether to include loss curves in the plot
        wandb_run: WandB run object for logging (optional)
        step: Training step/epoch for WandB logging (optional)
        
    Returns:
        dict: Summary statistics of gradients
    """
    if not gradient_logs:
        print("No gradient logs provided!")
        return {}
    
    # Filter logs based on component_names
    if component_names is not None:
        filtered_logs = {name: gradient_logs[name] for name in component_names if name in gradient_logs}
    else:
        filtered_logs = gradient_logs
    
    # Separate gradient logs from loss logs
    grad_logs = {k: v for k, v in filtered_logs.items() if not k.startswith('loss_')}
    loss_logs = {k: v for k, v in filtered_logs.items() if k.startswith('loss_')}
    
    # Determine subplot layout
    if show_losses and loss_logs:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        ax2 = None
    
    # Plot gradient magnitudes
    iterations = range(len(list(grad_logs.values())[0])) if grad_logs else []
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(grad_logs)))
    
    for i, (component, grad_values) in enumerate(grad_logs.items()):
        if len(grad_values) > 0:
            # Apply moving average smoothing if window_size > 1
            if window_size > 1 and len(grad_values) > window_size:
                smoothed_values = np.convolve(grad_values, np.ones(window_size)/window_size, mode='valid')
                smoothed_iterations = iterations[window_size-1:]
                ax1.plot(smoothed_iterations, smoothed_values, label=f'{component} (smoothed)', 
                        color=colors[i], linewidth=2)
                ax1.plot(iterations, grad_values, alpha=0.3, color=colors[i], linewidth=0.5)
            else:
                ax1.plot(iterations, grad_values, label=component, color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Gradient Magnitude (L2 Norm)')
    ax1.set_title('Gradient Magnitudes by Model Component')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot loss curves if requested
    if show_losses and loss_logs and ax2 is not None:
        loss_colors = plt.cm.Set1(np.linspace(0, 1, len(loss_logs)))
        
        for i, (loss_name, loss_values) in enumerate(loss_logs.items()):
            if len(loss_values) > 0:
                # Remove 'loss_' prefix from legend
                display_name = loss_name.replace('loss_', '')
                
                # Apply moving average smoothing
                if window_size > 1 and len(loss_values) > window_size:
                    smoothed_values = np.convolve(loss_values, np.ones(window_size)/window_size, mode='valid')
                    smoothed_iterations = iterations[window_size-1:]
                    ax2.plot(smoothed_iterations, smoothed_values, label=f'{display_name} (smoothed)', 
                            color=loss_colors[i], linewidth=2)
                    ax2.plot(iterations, loss_values, alpha=0.3, color=loss_colors[i], linewidth=0.5)
                else:
                    ax2.plot(iterations, loss_values, label=display_name, color=loss_colors[i], linewidth=2)
        
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Loss Value')
        ax2.set_title('Loss Components Over Time')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    
    # Save plot if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gradient magnitude plot saved to: {output_path}")
    
    # Calculate and return summary statistics
    stats = {}
    for component, grad_values in grad_logs.items():
        if len(grad_values) > 0:
            stats[component] = {
                'mean': np.mean(grad_values),
                'std': np.std(grad_values),
                'max': np.max(grad_values),
                'min': np.min(grad_values),
                'final': grad_values[-1] if len(grad_values) > 0 else 0,
                'trend': 'increasing' if len(grad_values) > 10 and grad_values[-1] > np.mean(grad_values[:10]) else 'decreasing'
            }
    
    plt.show()
    
    # Print summary
    print("\n=== Gradient Magnitude Summary ===")
    for component, component_stats in stats.items():
        print(f"\n{component}:")
        print(f"  Mean: {component_stats['mean']:.6f}")
        print(f"  Final: {component_stats['final']:.6f}")
        print(f"  Max: {component_stats['max']:.6f}")
        print(f"  Trend: {component_stats['trend']}")
    
    return stats


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
