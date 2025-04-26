import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm # Import colormap library

def visualize_trajectory(past_positions, future_positions, past_mask=None, future_mask=None, title="Trajectory", save_path=None, segment_idx=None, show_orientation=False, past_orientations=None, future_orientations=None, arrow_scale=0.2):
    """
    Visualize past (history) and future (ground truth) trajectory segments.
    
    Args:
        past_positions: Tensor of past positions [history_length, 3]
        future_positions: Tensor of future positions [future_length, 3]
        past_mask: Optional tensor mask [history_length]
        future_mask: Optional tensor mask [future_length]
        title: Plot title
        save_path: Path to save the figure
        segment_idx: Optional segment index for multi-segment objects
        show_orientation: Whether to show orientation arrows
        past_orientations: Tensor of past orientations [N, 3] (roll, pitch, yaw)
        future_orientations: Tensor of future orientations [M, 3] (roll, pitch, yaw)
        arrow_scale: Scale factor for orientation arrows
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Process Past Trajectory
    if isinstance(past_positions, torch.Tensor):
        past_positions = past_positions.detach().cpu().numpy()
    if past_mask is not None and isinstance(past_mask, torch.Tensor):
        past_mask = past_mask.detach().cpu().numpy()
        
    valid_past_indices = np.arange(past_positions.shape[0])
    if past_mask is not None:
        valid_past_indices = np.where(past_mask > 0.5)[0]
        if len(valid_past_indices) > 0:
            past_positions_valid = past_positions[valid_past_indices]
        else:
            past_positions_valid = np.empty((0, 3)) # Empty array if no valid past points
    else:
        past_positions_valid = past_positions
        
    num_valid_past = past_positions_valid.shape[0]

    # Process Future Trajectory
    if isinstance(future_positions, torch.Tensor):
        future_positions = future_positions.detach().cpu().numpy()
    if future_mask is not None and isinstance(future_mask, torch.Tensor):
        future_mask = future_mask.detach().cpu().numpy()

    valid_future_indices = np.arange(future_positions.shape[0])
    if future_mask is not None:
        valid_future_indices = np.where(future_mask > 0.5)[0]
        if len(valid_future_indices) > 0:
            future_positions_valid = future_positions[valid_future_indices]
        else:
             future_positions_valid = np.empty((0, 3)) # Empty array if no valid future points
    else:
        future_positions_valid = future_positions
        
    num_valid_future = future_positions_valid.shape[0]

    # Combine valid positions for plotting limits if needed, ensure not empty
    all_valid_positions = np.empty((0,3))
    if num_valid_past > 0:
        all_valid_positions = np.vstack((all_valid_positions, past_positions_valid))
    if num_valid_future > 0:
        all_valid_positions = np.vstack((all_valid_positions, future_positions_valid))
        
    if all_valid_positions.shape[0] == 0:
        # print("Warning: No valid points found in past or future trajectory.") # Can be noisy
        plt.close(fig)
        return
        
    # Plot Past Trajectory (History)
    if num_valid_past > 0:
        past_colors = cm.Blues(np.linspace(0.3, 1, num_valid_past)) # Use Blues colormap
        ax.plot(past_positions_valid[:, 0], past_positions_valid[:, 1], past_positions_valid[:, 2], 'gray', linestyle='-', linewidth=1, alpha=0.6, label='_nolegend_')
        ax.scatter(past_positions_valid[:, 0], past_positions_valid[:, 1], past_positions_valid[:, 2], c=past_colors, marker='o', s=25, label=f'History ({num_valid_past} pts)')
        # Highlight start of history
        ax.scatter(past_positions_valid[0, 0], past_positions_valid[0, 1], past_positions_valid[0, 2], c='lime', marker='o', s=100, label='Start History', edgecolors='black')

        # Add orientation arrows for past
        if show_orientation and past_orientations is not None:
            if isinstance(past_orientations, torch.Tensor):
                past_orientations = past_orientations.detach().cpu().numpy()
            for i in range(len(past_positions_valid)):
                roll, pitch, yaw = past_orientations[i]
                dx = arrow_scale * np.cos(yaw) * np.cos(pitch)
                dy = arrow_scale * np.sin(yaw) * np.cos(pitch)
                dz = arrow_scale * np.sin(pitch)
                
                ax.quiver(past_positions_valid[i, 0], past_positions_valid[i, 1], past_positions_valid[i, 2],
                         dx, dy, dz, color='r', arrow_length_ratio=0.3)

    # Plot Future Trajectory (Ground Truth)
    if num_valid_future > 0:
        future_colors = cm.Reds(np.linspace(0.3, 1, num_valid_future)) # Use Reds colormap
        ax.plot(future_positions_valid[:, 0], future_positions_valid[:, 1], future_positions_valid[:, 2], 'gray', linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')
        ax.scatter(future_positions_valid[:, 0], future_positions_valid[:, 1], future_positions_valid[:, 2], c=future_colors, marker='^', s=25, label=f'Future ({num_valid_future} pts)') # Use triangle markers
        # Highlight end of future
        ax.scatter(future_positions_valid[-1, 0], future_positions_valid[-1, 1], future_positions_valid[-1, 2], c='magenta', marker='o', s=100, label='End Future', edgecolors='black')

        # Add orientation arrows for future
        if show_orientation and future_orientations is not None:
            if isinstance(future_orientations, torch.Tensor):
                future_orientations = future_orientations.detach().cpu().numpy()
            for i in range(len(future_positions_valid)):
                roll, pitch, yaw = future_orientations[i]
                dx = arrow_scale * np.cos(yaw) * np.cos(pitch)
                dy = arrow_scale * np.sin(yaw) * np.cos(pitch)
                dz = arrow_scale * np.sin(pitch)
                
                ax.quiver(future_positions_valid[i, 0], future_positions_valid[i, 1], future_positions_valid[i, 2],
                         dx, dy, dz, color='b', arrow_length_ratio=0.3)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add segment information to title if available
    ax.set_title(f"{title} ({all_valid_positions.shape[0]} valid points)")
    ax.legend()
    
    # Add a color bar if points were plotted
    if ax.collections:
        cbar = fig.colorbar(ax.collections[-1], ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('Time Progression')
    
    # Save or show
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            # print(f"Saved trajectory visualization to {save_path}") # Can be noisy
        except Exception as e:
            print(f"Error saving trajectory visualization to {save_path}: {e}")
    else:
        plt.show()
    
    plt.close(fig)

def visualize_prediction(past_positions, future_positions_gt, future_positions_pred, 
                         past_mask=None, future_mask_gt=None, 
                         title="Prediction vs Ground Truth", save_path=None, segment_idx=None, show_orientation=False, past_orientations=None, future_orientations_gt=None, future_orientations_pred=None, arrow_scale=0.2):
    """
    Visualize past trajectory, ground truth future, and predicted future.
    Uses masks for past and ground truth future trajectories.

    Args:
        past_positions: Tensor or ndarray of past positions [history_length, 3]
        future_positions_gt: Tensor or ndarray of ground truth future positions [future_length, 3]
        future_positions_pred: Tensor or ndarray of predicted future positions [future_length, 3]
        past_mask: Optional mask for past positions [history_length]
        future_mask_gt: Optional mask for ground truth future positions [future_length]
        title: Plot title
        save_path: Path to save the figure
        segment_idx: Optional segment index for multi-segment objects
        show_orientation: Whether to show orientation arrows
        past_orientations: Tensor of past orientations [N, 3] (roll, pitch, yaw)
        future_orientations_gt: Tensor of ground truth future orientations [M, 3]
        future_orientations_pred: Tensor of predicted future orientations [M, 3]
        arrow_scale: Scale factor for orientation arrows
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- Process Past Trajectory with Mask ---
    if isinstance(past_positions, torch.Tensor):
        past_positions = past_positions.detach().cpu().numpy()
    if past_mask is not None and isinstance(past_mask, torch.Tensor):
        past_mask = past_mask.detach().cpu().numpy()
        
    valid_past_indices = np.arange(past_positions.shape[0])
    if past_mask is not None:
        valid_past_indices = np.where(past_mask > 0.5)[0]
        if len(valid_past_indices) > 0:
            past_positions_valid = past_positions[valid_past_indices]
        else:
            past_positions_valid = np.empty((0, 3)) 
    else:
        past_positions_valid = past_positions
    num_valid_past = past_positions_valid.shape[0]
    # ----------------------------------------

    # --- Process Future Trajectory (Ground Truth) with Mask ---
    if isinstance(future_positions_gt, torch.Tensor):
        future_positions_gt = future_positions_gt.detach().cpu().numpy()
    if future_mask_gt is not None and isinstance(future_mask_gt, torch.Tensor):
        future_mask_gt = future_mask_gt.detach().cpu().numpy()

    valid_future_gt_indices = np.arange(future_positions_gt.shape[0])
    if future_mask_gt is not None:
        valid_future_gt_indices = np.where(future_mask_gt > 0.5)[0]
        if len(valid_future_gt_indices) > 0:
            future_positions_gt_valid = future_positions_gt[valid_future_gt_indices]
        else:
             future_positions_gt_valid = np.empty((0, 3))
    else:
        future_positions_gt_valid = future_positions_gt
    num_valid_future_gt = future_positions_gt_valid.shape[0]
    # ----------------------------------------------------

    # --- Process Future Trajectory (Prediction) - No Mask --- 
    if isinstance(future_positions_pred, torch.Tensor):
        future_positions_pred = future_positions_pred.detach().cpu().numpy()
    num_future_pred = future_positions_pred.shape[0]
    # -----------------------------------------------------

    # Plot Past Trajectory (History) - Blue (using valid points)
    if num_valid_past > 0:
        past_colors = cm.Blues(np.linspace(0.3, 1, num_valid_past))
        ax.plot(past_positions_valid[:, 0], past_positions_valid[:, 1], past_positions_valid[:, 2], 'gray', linestyle='-', linewidth=1, alpha=0.6, label='_nolegend_')
        ax.scatter(past_positions_valid[:, 0], past_positions_valid[:, 1], past_positions_valid[:, 2], c=past_colors, marker='o', s=25, label=f'History ({num_valid_past} valid pts)')
        ax.scatter(past_positions_valid[0, 0], past_positions_valid[0, 1], past_positions_valid[0, 2], c='lime', marker='o', s=100, label='Start History', edgecolors='black')

        # Add orientation arrows for past
        if show_orientation and past_orientations is not None:
            if isinstance(past_orientations, torch.Tensor):
                past_orientations = past_orientations.detach().cpu().numpy()
            for i in range(len(past_positions_valid)):
                roll, pitch, yaw = past_orientations[i]
                dx = arrow_scale * np.cos(yaw) * np.cos(pitch)
                dy = arrow_scale * np.sin(yaw) * np.cos(pitch)
                dz = arrow_scale * np.sin(pitch)
                
                ax.quiver(past_positions_valid[i, 0], past_positions_valid[i, 1], past_positions_valid[i, 2],
                         dx, dy, dz, color='r', arrow_length_ratio=0.3)

    # Plot Future Trajectory (Ground Truth) - Green (using valid points)
    if num_valid_future_gt > 0:
        gt_colors = cm.Greens(np.linspace(0.3, 1, num_valid_future_gt))
        ax.plot(future_positions_gt_valid[:, 0], future_positions_gt_valid[:, 1], future_positions_gt_valid[:, 2], 'gray', linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')
        ax.scatter(future_positions_gt_valid[:, 0], future_positions_gt_valid[:, 1], future_positions_gt_valid[:, 2], c=gt_colors, marker='^', s=25, label=f'Ground Truth Future ({num_valid_future_gt} valid pts)')
        ax.scatter(future_positions_gt_valid[-1, 0], future_positions_gt_valid[-1, 1], future_positions_gt_valid[-1, 2], c='magenta', marker='o', s=100, label='End GT Future', edgecolors='black')

        # Add orientation arrows for ground truth future
        if show_orientation and future_orientations_gt is not None:
            if isinstance(future_orientations_gt, torch.Tensor):
                future_orientations_gt = future_orientations_gt.detach().cpu().numpy()
            for i in range(len(future_positions_gt_valid)):
                roll, pitch, yaw = future_orientations_gt[i]
                dx = arrow_scale * np.cos(yaw) * np.cos(pitch)
                dy = arrow_scale * np.sin(yaw) * np.cos(pitch)
                dz = arrow_scale * np.sin(pitch)
                
                ax.quiver(future_positions_gt_valid[i, 0], future_positions_gt_valid[i, 1], future_positions_gt_valid[i, 2],
                         dx, dy, dz, color='b', arrow_length_ratio=0.3)

    # Plot Future Trajectory (Prediction) - Red (plotting all points)
    if num_future_pred > 0:
        pred_colors = cm.Reds(np.linspace(0.3, 1, num_future_pred))
        ax.plot(future_positions_pred[:, 0], future_positions_pred[:, 1], future_positions_pred[:, 2], 'black', linestyle=':', linewidth=1, alpha=0.6, label='_nolegend_')
        ax.scatter(future_positions_pred[:, 0], future_positions_pred[:, 1], future_positions_pred[:, 2], c=pred_colors, marker='x', s=35, label=f'Predicted Future ({num_future_pred} pts)')
        ax.scatter(future_positions_pred[-1, 0], future_positions_pred[-1, 1], future_positions_pred[-1, 2], c='cyan', marker='o', s=100, label='End Predicted Future', edgecolors='black')

        # Add orientation arrows for predicted future
        if show_orientation and future_orientations_pred is not None:
            if isinstance(future_orientations_pred, torch.Tensor):
                future_orientations_pred = future_orientations_pred.detach().cpu().numpy()
            for i in range(len(future_positions_pred)):
                roll, pitch, yaw = future_orientations_pred[i]
                dx = arrow_scale * np.cos(yaw) * np.cos(pitch)
                dy = arrow_scale * np.sin(yaw) * np.cos(pitch)
                dz = arrow_scale * np.sin(pitch)
                
                ax.quiver(future_positions_pred[i, 0], future_positions_pred[i, 1], future_positions_pred[i, 2],
                         dx, dy, dz, color='g', arrow_length_ratio=0.3)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add segment information to title if available
    if segment_idx is not None:
        title = f"{title} (Segment {segment_idx})"
    ax.set_title(title)
    ax.legend()

    # Save or show
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            # print(f"Saved prediction visualization to {save_path}") # Can be noisy
        except Exception as e:
             print(f"Error saving prediction visualization to {save_path}: {e}")
    else:
        plt.show()

    plt.close(fig)

def visualize_full_trajectory(positions, attention_mask=None, point_cloud=None, title="Full Trajectory", save_path=None, segment_idx=None, show_orientation=False, orientations=None, arrow_scale=0.2, num_pc_points=10000):
    """
    Visualize a complete trajectory without past/future split, optionally with surrounding point cloud.
    
    Args:
        positions: Tensor or ndarray of positions [trajectory_length, 3]
        attention_mask: Optional mask [trajectory_length]
        point_cloud: Optional tensor or ndarray of point cloud points [N, 3]
        title: Plot title
        save_path: Path to save the figure
        segment_idx: Optional segment index for multi-segment objects
        show_orientation: Whether to show orientation arrows
        orientations: Tensor of orientations [N, 3] (roll, pitch, yaw)
        arrow_scale: Scale factor for orientation arrows
        num_pc_points: Max number of point cloud points to plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Process positions
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.detach().cpu().numpy()
    
    # Apply mask if provided
    valid_indices = np.arange(positions.shape[0])
    if attention_mask is not None:
        valid_indices = np.where(attention_mask > 0.5)[0]
        if len(valid_indices) > 0:
            positions_valid = positions[valid_indices]
        else:
            positions_valid = np.empty((0, 3))  # Empty array if no valid points
    else:
        positions_valid = positions
    
    num_valid_points = positions_valid.shape[0]
    
    if num_valid_points == 0:
        # print("Warning: No valid points found in trajectory.") # Can be noisy
        plt.close(fig)
        return
        
    # Render point cloud if provided
    if point_cloud is not None:
        # Convert point cloud to numpy
        if isinstance(point_cloud, torch.Tensor):
            point_cloud = point_cloud.detach().cpu().numpy()
            
        # Subsample point cloud if too many points to avoid rendering slowdown
        # max_pc_points = 10000  # Adjust if needed
        # if point_cloud.shape[0] > max_pc_points:
        #     indices = np.random.choice(point_cloud.shape[0], max_pc_points, replace=False)
        #     point_cloud = point_cloud[indices]
            
        # Plot the point cloud with blue color and transparency
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                  c='blue', s=1, alpha=0.3, label=f'Scene Points ({point_cloud.shape[0]})')
    
    # Create a colormap based on time progression
    point_colors = cm.viridis(np.linspace(0, 1, num_valid_points))
    
    # Plot the trajectory line
    ax.plot(positions_valid[:, 0], positions_valid[:, 1], positions_valid[:, 2], 
            'gray', linestyle='-', linewidth=1, alpha=0.6, label='_nolegend_')
    
    # Plot the trajectory points
    sc = ax.scatter(positions_valid[:, 0], positions_valid[:, 1], positions_valid[:, 2],
              c=point_colors, marker='o', s=30, label=f'Trajectory ({num_valid_points} pts)')
    
    # Highlight start and end
    ax.scatter(positions_valid[0, 0], positions_valid[0, 1], positions_valid[0, 2],
              c='lime', marker='o', s=100, label='Start', edgecolors='black')
    ax.scatter(positions_valid[-1, 0], positions_valid[-1, 1], positions_valid[-1, 2],
              c='magenta', marker='o', s=100, label='End', edgecolors='black')
    
    # Add orientation arrows if available
    if show_orientation and orientations is not None:
        for i in range(len(positions_valid)):
            roll, pitch, yaw = orientations[i]
            dx = arrow_scale * np.cos(yaw) * np.cos(pitch)
            dy = arrow_scale * np.sin(yaw) * np.cos(pitch)
            dz = arrow_scale * np.sin(pitch)
            
            # Use color gradient for arrows too
            arrow_color = point_colors[i]
            ax.quiver(positions_valid[i, 0], positions_valid[i, 1], positions_valid[i, 2],
                     dx, dy, dz, color=arrow_color, arrow_length_ratio=0.3)
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add segment information to title if available
    if segment_idx is not None:
        title = f"{title} (Segment {segment_idx})"
    ax.set_title(f"{title} ({num_valid_points} valid points)")
    ax.legend()
    
    # Add a color bar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=10) # Use scatter plot object for color bar
    cbar.set_label('Time Progression')
    
    # Save or show
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            # print(f"Saved full trajectory visualization to {save_path}") # Can be noisy
        except Exception as e:
            print(f"Error saving full trajectory visualization to {save_path}: {e}")
    else:
        plt.show()
    
    plt.close(fig)

def visualize_pointcloud(pointcloud, title="Point Cloud", save_path=None):
    """
    Visualize a point cloud.
    
    Args:
        pointcloud: Tensor or ndarray of points [N, 3]
        title: Plot title
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to numpy
    if isinstance(pointcloud, torch.Tensor):
        pointcloud = pointcloud.detach().cpu().numpy()
    
    # Sample points if too many to avoid clutter/slowdown
    max_pc_points = 10000 # Adjust if needed
    if pointcloud.shape[0] > max_pc_points:
        indices = np.random.choice(pointcloud.shape[0], max_pc_points, replace=False)
        pointcloud = pointcloud[indices]
    
    # Plot the point cloud
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c='blue', s=1, alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Save or show
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            # print(f"Saved pointcloud visualization to {save_path}") # Can be noisy
        except Exception as e:
            print(f"Error saving pointcloud visualization to {save_path}: {e}")
    else:
        plt.show()
    
    plt.close(fig) 