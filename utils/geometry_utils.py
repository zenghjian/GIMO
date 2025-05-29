#!/usr/bin/env python3
"""
Geometry utilities for GIMO ADT project.
Contains functions for rotation representation conversions and other geometric operations.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# --- 6D Rotation Representation Conversion Functions ---

def rotation_matrix_to_6d(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to 6D representation (first two columns).
    
    Args:
        rotation_matrix: 3x3 rotation matrix (numpy array)
        
    Returns:
        ndarray: 6D rotation representation [rx1, ry1, rz1, rx2, ry2, rz2]
    """
    # Extract first two columns of rotation matrix
    col1 = rotation_matrix[:, 0]  # First column
    col2 = rotation_matrix[:, 1]  # Second column
    
    # Concatenate into 6D representation
    r6d = np.concatenate([col1, col2])
    
    return r6d


def rotation_6d_to_matrix(r6d):
    """
    Convert 6D representation to rotation matrix using Gram-Schmidt process.
    
    Args:
        r6d: 6D rotation representation [rx1, ry1, rz1, rx2, ry2, rz2]
        
    Returns:
        ndarray: 3x3 rotation matrix
    """
    # Extract two columns
    x = r6d[:3]  # First column
    y = r6d[3:]  # Second column
    
    # Normalize first column
    x = x / np.linalg.norm(x)
    
    # Make second column orthogonal to first using Gram-Schmidt
    t = y - np.dot(x, y) * x
    y = t / np.linalg.norm(t)
    
    # Third column is cross product
    z = np.cross(x, y)
    
    # Stack to create rotation matrix
    return np.column_stack([x, y, z])


def rotation_6d_to_matrix_torch(r6d):
    """
    PyTorch version: Convert 6D representation to rotation matrix.
    
    Args:
        r6d: 6D rotation representation tensor [..., 6]
        
    Returns:
        torch.Tensor: rotation matrix tensor [..., 3, 3]
    """
    # Extract two columns
    x = r6d[..., :3]  # First column
    y = r6d[..., 3:]  # Second column
    
    # Normalize first column
    x = x / torch.norm(x, dim=-1, keepdim=True)
    
    # Make second column orthogonal to first using Gram-Schmidt
    t = y - torch.sum(x * y, dim=-1, keepdim=True) * x
    y = t / torch.norm(t, dim=-1, keepdim=True)
    
    # Third column is cross product
    z = torch.cross(x, y, dim=-1)
    
    # Stack to create rotation matrix
    return torch.stack([x, y, z], dim=-1)


def rotation_matrix_to_6d_torch(rotation_matrix):
    """
    PyTorch version: Convert rotation matrix to 6D representation.
    
    Args:
        rotation_matrix: rotation matrix tensor [..., 3, 3]
        
    Returns:
        torch.Tensor: 6D rotation representation tensor [..., 6]
    """
    # Extract first two columns
    col1 = rotation_matrix[..., :, 0]  # First column
    col2 = rotation_matrix[..., :, 1]  # Second column
    
    # Concatenate into 6D representation
    return torch.cat([col1, col2], dim=-1)


def convert_6d_to_euler(rotations):
    """
    Convert 6D rotation representation to Euler angles for visualization.
    Supports both numpy arrays and torch tensors.
    
    Args:
        rotations: 6D rotation representation [..., 6] or [..., 3] if already Euler
        
    Returns:
        ndarray: Euler angles in radians [roll, pitch, yaw] format
    """
    if rotations is None:
        return None
    
    # Convert to numpy if tensor
    if isinstance(rotations, torch.Tensor):
        rotations = rotations.detach().cpu().numpy()
    
    if rotations.shape[-1] == 3:
        # Already Euler angles
        return rotations
    elif rotations.shape[-1] == 6:
        # Convert 6D to Euler
        result = []
        for i in range(rotations.shape[0]):
            r6d = rotations[i]
            
            # Convert 6D to rotation matrix first
            rotation_matrix = rotation_6d_to_matrix(r6d)
            
            # Convert rotation matrix to Euler angles
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + 
                        rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6
            
            if not singular:
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = 0
            
            result.append([roll, pitch, yaw])
        return np.array(result)
    else:
        print(f"Warning: Unexpected rotation dimension {rotations.shape[-1]}. Expected 3 or 6.")
        return None


def euler_to_rotation_matrix(euler_angles):
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        euler_angles: Euler angles [roll, pitch, yaw] in radians
        
    Returns:
        ndarray: 3x3 rotation matrix
    """
    if isinstance(euler_angles, torch.Tensor):
        euler_angles = euler_angles.detach().cpu().numpy()
    
    roll, pitch, yaw = euler_angles
    
    # Create rotation matrix from Euler angles
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix (ZYX convention)
    R = R_z @ R_y @ R_x
    return R


def euler_to_6d(euler_angles):
    """
    Convert Euler angles to 6D rotation representation.
    
    Args:
        euler_angles: Euler angles [roll, pitch, yaw] in radians
        
    Returns:
        ndarray: 6D rotation representation
    """
    rotation_matrix = euler_to_rotation_matrix(euler_angles)
    return rotation_matrix_to_6d(rotation_matrix)


# --- Coordinate Transformation Functions ---

def transform_coords_for_visualization(tensor_3d):
    """
    Apply coordinate transformation (x, y, z) -> (x, -z, y) for visualization.
    
    Args:
        tensor_3d: Input tensor with 3D coordinates [..., 3]
        
    Returns:
        torch.Tensor: Transformed tensor with same shape
    """
    if tensor_3d is None or tensor_3d.numel() == 0:
        return tensor_3d
    
    # Ensure it's a tensor
    if not isinstance(tensor_3d, torch.Tensor):
        tensor_3d = torch.tensor(tensor_3d)

    if tensor_3d.shape[-1] != 3:
        return tensor_3d

    x = tensor_3d[..., 0]
    y = tensor_3d[..., 1]
    z = tensor_3d[..., 2]
    
    transformed_tensor = torch.stack((x, -z, y), dim=-1)
    return transformed_tensor


# --- Utility Functions ---

def calculate_dynamic_arrow_scale(positions, min_scale=0.05, max_scale=0.5, percentage=0.05):
    """
    Calculate dynamic arrow scale for orientation visualization based on trajectory extent.
    
    Args:
        positions: Position array [..., 3]
        min_scale: Minimum arrow scale
        max_scale: Maximum arrow scale  
        percentage: Percentage of trajectory range to use for scale
        
    Returns:
        float: Calculated arrow scale
    """
    if positions is None or len(positions) == 0:
        return min_scale
        
    # Convert to numpy if needed
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
        
    if positions.shape[0] <= 1:
        return min_scale
        
    # Calculate range
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    ranges = max_coords - min_coords
    max_range = np.max(ranges) if ranges.size > 0 else 0
    
    if max_range > 1e-6:
        calculated_scale = max_range * percentage
        return np.clip(calculated_scale, min_scale, max_scale)
    else:
        return min_scale 