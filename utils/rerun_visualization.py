#!/usr/bin/env python3
"""
Rerun Visualization Utilities for GIMO ADT
------------------------------------------
This module provides functions to visualize trajectory data using Rerun.
It supports visualizing past trajectories, ground truth futures, predictions,
and point clouds.
"""

import os
import numpy as np
import torch
import subprocess
import sys
from typing import Dict, List, Tuple, Optional, Union, Any

# Check for availability of rerun visualization
try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    print("Warning: rerun module not available, some visualization features will be disabled.")
    HAS_RERUN = False

def initialize_rerun(recording_name="gimo_adt_evaluation", spawn=True, output_dir=None) -> Tuple[bool, Optional[str]]:
    """
    Initialize Rerun for visualization.
    
    Args:
        recording_name: Name of the Rerun recording. This will be used as the application ID.
        spawn: Whether to spawn a new Rerun viewer. For evaluation, this is often set to False
               if the goal is to save the recording to a file without viewing it immediately.
        output_dir: Directory to save .rrd files. If provided, Rerun is configured to save
                    the recording directly to a file. If None, the recording is kept in memory.
        
    Returns:
        Tuple[bool, Optional[str]]: 
            - bool: Whether Rerun was successfully initialized.
            - Optional[str]: The target path for the .rrd file if output_dir was provided, else None.
    """
    if not HAS_RERUN:
        print("Warning: rerun module not available. Cannot initialize.")
        return False, None
    
    out_path: Optional[str] = None # Initialize out_path to None

    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{recording_name}.rrd")
            rr.init(recording_name, spawn=spawn)
            print(f"Initialized Rerun (older version or fallback) with recording name: {recording_name}. Saving to {out_path} may need explicit 'rr.save()' if not handled by init.")
            return True, out_path 
        else:
            rr.init(recording_name, spawn=spawn)
            return True, None
        
    except Exception as e:
        print(f"Error initializing Rerun: {e}")
        return False, out_path

def downsample_point_cloud(point_cloud, downsample_factor=10):
    """
    Downsample a point cloud by selecting every nth point.
    
    Args:
        point_cloud: Point cloud data of shape [N, 3] or [N, 4] if it includes intensity
        downsample_factor: Factor by which to downsample (1 = no downsampling)
        
    Returns:
        Downsampled point cloud
    """
    if downsample_factor <= 1:
        return point_cloud
    
    if point_cloud.shape[0] > 1000000:
        num_samples = point_cloud.shape[0] // downsample_factor
        indices = np.random.choice(point_cloud.shape[0], size=num_samples, replace=False)
        return point_cloud[indices]
    
    return point_cloud[::downsample_factor]

def extract_trajectory_specific_point_cloud(point_cloud, trajectory, radius=1.0, max_points=100000):
    """
    Extract points from a point cloud that are near a trajectory.
    
    Args:
        point_cloud: Point cloud data of shape [N, 3] or [N, 4] if it includes intensity
        trajectory: Trajectory positions of shape [T, 3]
        radius: Radius around trajectory points to include
        max_points: Maximum number of points to return
        
    Returns:
        Point cloud data near the trajectory
    """
    if point_cloud is None or point_cloud.shape[0] == 0 or trajectory.shape[0] == 0:
        return None
    
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach().cpu().numpy()
    
    pc_positions = point_cloud[:, :3]
    
    near_points_indices = []
    for traj_point in trajectory:
        distances = np.linalg.norm(pc_positions - traj_point, axis=1)
        near_indices = np.where(distances < radius)[0]
        near_points_indices.extend(near_indices)
    
    near_points_indices = list(set(near_points_indices))
    
    if len(near_points_indices) > max_points:
        near_points_indices = np.random.choice(near_points_indices, size=max_points, replace=False)
    
    return point_cloud[near_points_indices]

def visualize_trajectory_rerun(
    past_positions, 
    future_positions_gt=None, 
    future_positions_pred=None,
    past_mask=None, 
    future_mask_gt=None,
    point_cloud=None,
    past_orientations=None,
    future_orientations_gt=None,
    future_orientations_pred=None,
    object_name="object",
    sequence_name="sequence",
    segment_idx=None,
    arrow_length=0.3,
    line_width=0.02,
    point_size=0.03
):
    """
    Visualize trajectory data using Rerun.
    
    Args:
        past_positions: Past trajectory positions [history_length, 3]
        future_positions_gt: Ground truth future positions [future_length, 3]
        future_positions_pred: Predicted future positions [future_length, 3]
        past_mask: Optional mask for past positions [history_length]
        future_mask_gt: Optional mask for ground truth future positions [future_length]
        point_cloud: Optional point cloud data [N, 3] or [N, 4]
        past_orientations: Past orientations [history_length, 3] (roll, pitch, yaw)
        future_orientations_gt: Ground truth future orientations [future_length, 3]
        future_orientations_pred: Predicted future orientations [future_length, 3]
        object_name: Name/ID of the object, used to create a unique path in Rerun.
        sequence_name: Name of the sequence, used in the Rerun path.
        segment_idx: Optional segment index, used in the Rerun path.
        arrow_length: Length of orientation arrows.
        line_width: Width of trajectory lines.
        point_size: Size of trajectory points.
    """
    if not HAS_RERUN:
        return False
    
    vis_path = f"trajectories/{sequence_name}"
    if segment_idx is not None:
        vis_path += f"/segment_{segment_idx}"
    vis_path += f"/{object_name}"
    
    if isinstance(past_positions, torch.Tensor):
        past_positions = past_positions.detach().cpu().numpy()
    if isinstance(future_positions_gt, torch.Tensor):
        future_positions_gt = future_positions_gt.detach().cpu().numpy()
    if isinstance(future_positions_pred, torch.Tensor):
        future_positions_pred = future_positions_pred.detach().cpu().numpy()
    if isinstance(past_mask, torch.Tensor):
        past_mask = past_mask.detach().cpu().numpy()
    if isinstance(future_mask_gt, torch.Tensor):
        future_mask_gt = future_mask_gt.detach().cpu().numpy()
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    elif point_cloud is not None:
        pass
    else:
        pass

    if isinstance(past_orientations, torch.Tensor):
        past_orientations = past_orientations.detach().cpu().numpy()
    if isinstance(future_orientations_gt, torch.Tensor):
        future_orientations_gt = future_orientations_gt.detach().cpu().numpy()
    if isinstance(future_orientations_pred, torch.Tensor):
        future_orientations_pred = future_orientations_pred.detach().cpu().numpy()
    
    if past_mask is not None:
        valid_past_indices = np.where(past_mask > 0.5)[0]
        if len(valid_past_indices) > 0:
            past_positions_valid = past_positions[valid_past_indices]
            past_orientations_valid = past_orientations[valid_past_indices] if past_orientations is not None else None
        else:
            past_positions_valid = np.empty((0, 3))
            past_orientations_valid = np.empty((0, 3)) if past_orientations is not None else None
    else:
        past_positions_valid = past_positions
        past_orientations_valid = past_orientations

    if future_positions_gt is not None and future_mask_gt is not None:
        valid_future_gt_indices = np.where(future_mask_gt > 0.5)[0]
        if len(valid_future_gt_indices) > 0:
            future_positions_gt_valid = future_positions_gt[valid_future_gt_indices]
            future_orientations_gt_valid = future_orientations_gt[valid_future_gt_indices] if future_orientations_gt is not None else None
        else:
            future_positions_gt_valid = np.empty((0, 3))
            future_orientations_gt_valid = np.empty((0, 3)) if future_orientations_gt is not None else None
    else:
        future_positions_gt_valid = future_positions_gt
        future_orientations_gt_valid = future_orientations_gt

    future_positions_pred_valid = future_positions_pred
    future_orientations_pred_valid = future_orientations_pred
    
    try:
        current_animation_step = 0 # Initialize animation_step counter first

        # Log point cloud once, at the first step of the "animation_step" timeline
        if point_cloud is not None and len(point_cloud) > 0:
            rr.set_time_sequence("animation_step", current_animation_step) # Ensure PC is on the animation timeline
            point_cloud_colors = np.ones((point_cloud.shape[0], 4), dtype=np.float32) * [0.7, 0.7, 0.7, 0.3]
            try:
                rr.log(f"{vis_path}/point_cloud", 
                      rr.Points3D(
                          positions=point_cloud[:, :3],
                          colors=point_cloud_colors,
                          radii=0.005 # Consider making this configurable or relative to scene scale
                      ))
            except Exception as e:
                print(f"Error logging point cloud: {e}")
        
        # current_animation_step is still 0 here. 
        # Trajectory animations will start their loops using this current_animation_step = 0,
        # and then increment it inside their respective loops.
        
        # --- Animate Past Trajectory ---
        if past_positions_valid.shape[0] > 0:
            for t in range(past_positions_valid.shape[0]):
                rr.set_time_sequence("animation_step", current_animation_step)
                # Log growing past path (line)
                rr.log(f"{vis_path}/trajectory/past_path",
                       rr.LineStrips3D(
                           [past_positions_valid[0:t+1].tolist()],
                           colors=[[0.6, 0.6, 1, 0.7]], # Lighter Blue
                           radii=line_width
                       ))
                # Log growing past points
                rr.log(f"{vis_path}/trajectory/past_points",
                       rr.Points3D(
                           positions=past_positions_valid[0:t+1],
                           colors=[0.3, 0.3, 1, 0.5], 
                           radii=point_size
                       ))
                current_animation_step += 1
        
        # Define connection point from past trajectory
        connection_point_np = past_positions_valid[-1] if past_positions_valid.shape[0] > 0 else None

        # --- Animate Future Ground Truth Trajectory ---
        if future_positions_gt_valid is not None and future_positions_gt_valid.shape[0] > 0:
            # Path for GT trajectory, possibly connected to past
            gt_path_points_for_strip = []
            if connection_point_np is not None:
                gt_path_points_for_strip.append(connection_point_np.tolist())

            for t in range(future_positions_gt_valid.shape[0]):
                rr.set_time_sequence("animation_step", current_animation_step)
                
                # Keep past trajectory (line and points) visible
                if past_positions_valid.shape[0] > 0:
                    rr.log(f"{vis_path}/trajectory/past_path",
                           rr.LineStrips3D(
                               [past_positions_valid.tolist()],
                               colors=[[0.6, 0.6, 1, 0.7]], # Lighter Blue
                               radii=line_width
                           ))
                    rr.log(f"{vis_path}/trajectory/past_points",
                           rr.Points3D(
                               positions=past_positions_valid,
                               colors=[0.3, 0.3, 1, 0.5], 
                               radii=point_size
                           ))

                # Construct current segment of GT path (line)
                current_gt_segment_list = future_positions_gt_valid[0:t+1].tolist()
                
                if connection_point_np is not None:
                    points_to_log_gt = [connection_point_np.tolist()] + current_gt_segment_list
                else:
                    points_to_log_gt = current_gt_segment_list
                
                if points_to_log_gt: # Ensure there are points to log
                    rr.log(f"{vis_path}/trajectory/gt_path",
                           rr.LineStrips3D(
                               [points_to_log_gt],
                               colors=[[0.6, 1, 0.6, 0.7]], # Lighter Green
                               radii=line_width
                           ))
                
                # Log growing GT points
                if future_positions_gt_valid[0:t+1].shape[0] > 0:
                    rr.log(f"{vis_path}/trajectory/gt_points",
                           rr.Points3D(
                               positions=future_positions_gt_valid[0:t+1],
                               colors=[0.3, 1, 0.3, 0.5], # Solid Green
                               radii=point_size
                           ))
                current_animation_step += 1
        
        # --- Animate Future Predicted Trajectory ---
        if future_positions_pred_valid is not None and future_positions_pred_valid.shape[0] > 0:
            # Path for Pred trajectory, possibly connected to past
            pred_path_points_for_strip = []
            if connection_point_np is not None:
                pred_path_points_for_strip.append(connection_point_np.tolist())

            # Determine full GT path if it existed, to keep it visible
            full_gt_path_list = []
            if future_positions_gt_valid is not None and future_positions_gt_valid.shape[0] > 0:
                if connection_point_np is not None:
                    full_gt_path_list = [connection_point_np.tolist()] + future_positions_gt_valid.tolist()
                else:
                    full_gt_path_list = future_positions_gt_valid.tolist()

            for t in range(future_positions_pred_valid.shape[0]):
                rr.set_time_sequence("animation_step", current_animation_step)

                # Keep past trajectory (line and points) visible
                if past_positions_valid.shape[0] > 0:
                    rr.log(f"{vis_path}/trajectory/past_path",
                           rr.LineStrips3D(
                               [past_positions_valid.tolist()],
                               colors=[[0.6, 0.6, 1, 0.7]], # Lighter Blue
                               radii=line_width
                           ))
                    rr.log(f"{vis_path}/trajectory/past_points",
                           rr.Points3D(
                               positions=past_positions_valid,
                               colors=[0.3, 0.3, 1, 0.5], # Solid Blue
                               radii=point_size
                           ))
                
                # Keep GT trajectory (line and points) visible if it existed
                if full_gt_path_list:
                     rr.log(f"{vis_path}/trajectory/gt_path",
                           rr.LineStrips3D(
                               [full_gt_path_list],
                               colors=[[0.6, 1, 0.6, 0.7]], # Lighter Green
                               radii=line_width
                           ))
                     if future_positions_gt_valid is not None and future_positions_gt_valid.shape[0] > 0:
                         rr.log(f"{vis_path}/trajectory/gt_points",
                                rr.Points3D(
                                    positions=future_positions_gt_valid,
                                    colors=[0.3, 1, 0.3, 0.5], # Solid Green
                                    radii=point_size
                                ))

                # Construct current segment of Pred path (line)
                current_pred_segment_list = future_positions_pred_valid[0:t+1].tolist()

                if connection_point_np is not None:
                    points_to_log_pred = [connection_point_np.tolist()] + current_pred_segment_list
                else:
                    points_to_log_pred = current_pred_segment_list
                
                if points_to_log_pred: # Ensure there are points to log
                    rr.log(f"{vis_path}/trajectory/pred_path",
                           rr.LineStrips3D(
                               [points_to_log_pred],
                               colors=[[1, 0.6, 0.6, 0.7]], # Lighter Red
                               radii=line_width
                           ))
                
                # Log growing Pred points
                if future_positions_pred_valid[0:t+1].shape[0] > 0:
                    rr.log(f"{vis_path}/trajectory/pred_points",
                           rr.Points3D(
                               positions=future_positions_pred_valid[0:t+1],
                               colors=[1, 0.3, 0.3, 0.5], # Solid Red
                               radii=point_size
                           ))
                current_animation_step += 1
            
    except Exception as e:
        print(f"Error in Rerun visualization: {e}")
        return False
        
    return True

def save_rerun_recording(output_path, vis_path=None):
    """
    Save a Rerun recording to a file.
    
    NOTE: In modern Rerun versions (e.g., 0.9.0+), if Rerun was initialized with the
    `to_file` parameter (as done in `initialize_rerun` when `output_dir` is provided),
    the recording is ALREADY being saved to the specified .rrd file. Calling `rr.save()`
    explicitly might be redundant or could even cause issues in such cases.
    This function is retained for potential backward compatibility or for scenarios
    where Rerun was initialized without `to_file` (e.g., in-memory recording) and
    now needs to be saved. The `evaluate_gimo_adt.py` script relies on the `to_file`
    mechanism in `initialize_rerun` and does NOT call this `save_rerun_recording` function.
    
    Args:
        output_path: Path to save the recording (e.g., "my_recording.rrd").
        vis_path: Specific visualization path to save. If None, attempts to save
                  the entire current Rerun recording. This argument is often not
                  directly supported by `rr.save()` in the way one might expect
                  for partial saves; `rr.save()` typically saves the whole recording.
                  The Rerun CLI `save` command also saves the whole recording.
        
    Returns:
        bool: Whether the recording was successfully saved or likely already saved.
    """
    if not HAS_RERUN:
        print("Warning: rerun module not available. Cannot save recording.")
        return False
    
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory for output_path {output_path}: {e}")
        return False

    try:
        if hasattr(rr, 'save') and callable(rr.save):
            rr.save(output_path)
            if os.path.exists(output_path):
                return True
        else:
            print("rr.save() function not available in this Rerun version or context.")
            
    except Exception as e_py_save:
        print(f"Error or issue using Python rr.save('{output_path}'): {e_py_save}")
        print("This might be expected if 'to_file' was used during rr.init().")

    if os.path.exists(output_path):
        print(f"Confirmed: Rerun recording file {output_path} exists (likely saved during initialization).")
        return True
        
    print(f"Attempting to use Rerun command-line tool to save recording as a fallback for {output_path}...")
    try:
        rerun_cmd = "rerun"
        result = subprocess.run([rerun_cmd, "--version"], 
                               capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print(f"Found rerun CLI: {result.stdout.strip()}")
            save_args = [rerun_cmd, "save", output_path]
            if vis_path:
                print(f"Warning: 'vis_path' argument ('{vis_path}') is not directly used by 'rerun save' CLI command.")
            
            print(f"Executing Rerun CLI save: {' '.join(save_args)}")
            save_result = subprocess.run(save_args, capture_output=True, text=True, check=False)
            
            if save_result.returncode == 0:
                print(f"Successfully saved recording to {output_path} using rerun CLI (stdout: {save_result.stdout.strip()})")
                return True
            else:
                print(f"Error using rerun CLI 'save' (return code {save_result.returncode}):")
                print(f"  Stdout: {save_result.stdout.strip()}")
                print(f"  Stderr: {save_result.stderr.strip()}")
        else:
            print("rerun CLI not available or --version command returned error.")
            
    except FileNotFoundError:
        print("rerun CLI command not found. Ensure Rerun is installed and in PATH for CLI fallback.")
    except Exception as cmd_e:
        print(f"Error during Rerun CLI save attempt: {cmd_e}")
            
    if not os.path.exists(output_path):
        print(f"Warning: Could not save or confirm Rerun recording at {output_path} after all attempts.")
        print("If 'output_dir' was specified during 'initialize_rerun', the recording should have been saved there directly.")
        print(f"Please check if '{output_path}' (or a similar file in the init 'output_dir') was created during 'rr.init()'.")
        return False
    
    return os.path.exists(output_path) 