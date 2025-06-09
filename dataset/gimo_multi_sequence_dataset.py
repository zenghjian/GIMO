#!/usr/bin/env python3
# Dataset class for loading multiple Aria Digital Twin sequences for GIMO
# Simplified version that doesn't use the PointNet encoder

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import traceback
from typing import Dict, List, Tuple, Optional, Union
import sys


from dataset.gimo_adt_trajectory_dataset import GimoAriaDigitalTwinTrajectoryDataset


class GIMOMultiSequenceDataset(Dataset):
    """Dataset for loading trajectories from multiple Aria Digital Twin sequences for GIMO model."""
    
    def __init__(
        self, 
        sequence_paths: List[str],
        config=None,  # Added config parameter
        trajectory_length: int = 100, 
        history_fraction: float = 0.375,
        skip_frames: int = 5,
        max_objects: Optional[int] = None,
        device_num: int = 0,
        transform=None,
        load_pointcloud: bool = True,
        pointcloud_subsample: int = 1,  # Changed to 1 to use full resolution
        min_motion_threshold: float = 1.0,
        min_motion_percentile: float = 0.0,
        use_displacements: bool = False,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        normalize_data: bool = True,
        trajectory_pointcloud_radius: float = 0.5,  # Added parameter for trajectory filtering
        max_bboxes: int = 400,  # Added parameter for bbox handling
    ):
        """
        Initialize the multi-sequence dataset for GIMO.
        
        Args:
            sequence_paths: List of paths to ADT sequences
            config: Optional configuration object with dataset parameters
            trajectory_length: Length of trajectories to extract (if config not provided)
            history_fraction: Fraction of trajectory_length used for history (if config not provided)
            skip_frames: Number of frames to skip between samples (if config not provided)
            max_objects: Maximum number of objects per sequence (None for all)
            device_num: Device number for ADT data provider
            transform: Optional transform to apply to trajectories
            load_pointcloud: Whether to load the MPS pointcloud (if config not provided)
            pointcloud_subsample: Subsample factor for pointcloud (if config not provided)
            min_motion_threshold: Minimum total path length threshold in meters (if config not provided)
            min_motion_percentile: Filter trajectories below this percentile of motion (if config not provided)
            use_displacements: Whether to use displacements instead of absolute positions (if config not provided)
            use_cache: Whether to use caching (if config not provided)
            cache_dir: Directory for caching trajectory data (if config not provided)
            normalize_data: Whether to normalize data using scene bounds (if config not provided)
            trajectory_pointcloud_radius: Radius around trajectory to collect points (meters)
            max_bboxes: Maximum number of bounding boxes for consistent batching (if config not provided)
        """
        self.sequence_paths = sequence_paths
        
        # Use config values if provided, otherwise use the explicitly passed parameters
        if config is not None:
            self.trajectory_length = getattr(config, 'trajectory_length', trajectory_length)
            self.history_fraction = getattr(config, 'history_fraction', history_fraction)
            self.skip_frames = getattr(config, 'skip_frames', skip_frames)
            self.load_pointcloud = getattr(config, 'load_pointcloud', load_pointcloud)
            self.pointcloud_subsample = getattr(config, 'pointcloud_subsample', pointcloud_subsample)
            self.min_motion_threshold = getattr(config, 'min_motion_threshold', min_motion_threshold)
            self.min_motion_percentile = getattr(config, 'min_motion_percentile', min_motion_percentile)
            self.use_displacements = getattr(config, 'use_displacements', use_displacements)
            self.use_cache = getattr(config, 'use_cache', use_cache)
            self.normalize_data = getattr(config, 'normalize_data', normalize_data)
            self.trajectory_pointcloud_radius = getattr(config, 'trajectory_pointcloud_radius', trajectory_pointcloud_radius)
            self.force_use_cache = getattr(config, 'force_use_cache', False) # Get force_use_cache from config
            self.max_bboxes = getattr(config, 'max_bboxes', max_bboxes)
            # For cache_dir, don't use config directly - it will be set below from save_path if provided
        else:
            self.trajectory_length = trajectory_length
            self.history_fraction = history_fraction
            self.skip_frames = skip_frames
            self.load_pointcloud = load_pointcloud
            self.pointcloud_subsample = pointcloud_subsample
            self.min_motion_threshold = min_motion_threshold
            self.min_motion_percentile = min_motion_percentile
            self.use_displacements = use_displacements
            self.use_cache = use_cache
            self.normalize_data = normalize_data
            self.trajectory_pointcloud_radius = trajectory_pointcloud_radius
            self.max_bboxes = max_bboxes
        
        # These parameters don't typically come from config
        self.max_objects = max_objects
        self.device_num = device_num
        self.transform = transform
        
        
        # --- Determine Cache Directory ---
        effective_cache_dir = None
        if cache_dir is not None:
            effective_cache_dir = cache_dir
            print(f"Using explicitly provided cache directory: {effective_cache_dir}")
        elif config is not None and getattr(config, 'global_cache_dir', None):
            effective_cache_dir = config.global_cache_dir
            print(f"Using global cache directory from config: {effective_cache_dir}")
        elif config is not None and hasattr(config, 'save_path'):
            effective_cache_dir = os.path.join(config.save_path, 'trajectory_cache')
            print(f"Using experiment-specific cache directory derived from save_path: {effective_cache_dir}")
        else:
            effective_cache_dir = './trajectory_cache'  # Default fallback
            print(f"Using default cache directory: {effective_cache_dir}")
        self.cache_dir = effective_cache_dir
        # -----------------------------------
        
        # Calculate history and future lengths based on fraction
        # Use floor for history length to ensure it's an integer
        self.history_length = int(np.floor(self.trajectory_length * self.history_fraction))
        # Ensure future length makes the total equal to trajectory_length
        self.future_length = self.trajectory_length - self.history_length
        print(f"Trajectory split: History={self.history_length}, Future={self.future_length} (based on fraction {self.history_fraction:.3f})")
        
        # This will store all individual datasets
        self.individual_datasets = []
        
        # Map from global index to (dataset_index, local_index)
        self.index_map = []
        
        # Track which sequences were successfully loaded
        self.loaded_sequences = []
        
        # Create cache directory if it doesn't exist and we're using cache
        if self.use_cache and not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir)
                print(f"Created cache directory: {self.cache_dir}")
            except Exception as e:
                print(f"Warning: Could not create cache directory: {e}")
                self.use_cache = False
                
        # Load all sequences
        self._load_sequences()
        
        print(f"GIMOMultiSequenceDataset initialized with {len(self.loaded_sequences)} sequences")
        print(f"Total trajectories: {len(self)}")
    
    def _load_sequences(self):
        """Load all specified sequences."""
        print(f"Loading {len(self.sequence_paths)} ADT sequences...")
        
        for i, seq_path in enumerate(self.sequence_paths):
            try:
                print(f"Loading sequence {i+1}/{len(self.sequence_paths)}: {os.path.basename(seq_path)}")
                
                # Create a dataset for this sequence with parameters from this instance
                dataset = GimoAriaDigitalTwinTrajectoryDataset(
                    sequence_path=seq_path,
                    trajectory_length=self.trajectory_length,
                    skip_frames=self.skip_frames,
                    max_objects=self.max_objects,
                    device_num=self.device_num,
                    transform=self.transform,
                    load_pointcloud=self.load_pointcloud,
                    pointcloud_subsample=self.pointcloud_subsample,
                    min_motion_threshold=self.min_motion_threshold,
                    min_motion_percentile=self.min_motion_percentile,
                    use_displacements=self.use_displacements,
                    use_cache=self.use_cache,
                    cache_dir=self.cache_dir,
                    detect_motion_segments=getattr(self, 'detect_motion_segments', True),
                    motion_velocity_threshold=getattr(self, 'motion_velocity_threshold', 0.05),
                    min_segment_frames=getattr(self, 'min_segment_frames', 5),
                    max_stationary_frames=getattr(self, 'max_stationary_frames', 3),
                    normalize_data=self.normalize_data,
                    trajectory_pointcloud_radius=self.trajectory_pointcloud_radius,
                    force_use_cache=self.force_use_cache,  # Pass force_use_cache parameter
                    max_bboxes=self.max_bboxes  # Pass max_bboxes parameter
                )
                
                # If the dataset has trajectories, add it to our collection
                if len(dataset) > 0:
                    self.individual_datasets.append(dataset)
                    self.loaded_sequences.append(seq_path)
                    
                    # Update the index map
                    dataset_idx = len(self.individual_datasets) - 1
                    for local_idx in range(len(dataset)):
                        self.index_map.append((dataset_idx, local_idx))
                    
                    print(f"  Added {len(dataset)} trajectories from sequence")
                else:
                    print(f"  Skipping sequence - no trajectories found")
                    
            except Exception as e:
                print(f"Error loading sequence {seq_path}: {e}")
                traceback.print_exc()
    
    def __len__(self):
        """Get the total number of trajectories across all datasets."""
        return len(self.index_map)
    
    def __getitem__(self, idx):
        """
        Get a trajectory sample by global index.
        
        Args:
            idx: Global index of the trajectory
            
        Returns:
            dict: Trajectory data
        """
        if idx < 0 or idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of range [0, {len(self.index_map)-1}]")
        
        # Look up which dataset and local index to use
        dataset_idx, local_idx = self.index_map[idx]
        
        # Get the item from the individual dataset
        sample = self.individual_datasets[dataset_idx][local_idx]
        # Also get the original item to access full metadata if needed
        original_traj_item = self.individual_datasets[dataset_idx].trajectories[local_idx]
        
        # Add the sequence info
        sequence_path = self.loaded_sequences[dataset_idx]
        sample['sequence_path'] = sequence_path
        sample['sequence_name'] = os.path.basename(sequence_path)
        sample['dataset_idx'] = dataset_idx # Ensure dataset_idx is in the sample for collate_fn

        # Add category to the sample
        sample['object_category'] = original_traj_item.get('metadata', {}).get('category', 'unknown')
        
        # --- Perform Past/Future Split Here ---
        # Check if we have poses (9D) or positions (3D) - handle both for backward compatibility
        if 'poses' in sample:
            # Rename to full_ versions and remove original keys
            sample['full_poses'] = sample.pop('poses')
            sample['full_attention_mask'] = sample.pop('attention_mask')
            
            # Make sure positions/rotations are also available
            if 'positions' not in sample:
                sample['full_positions'] = sample['full_poses'][:, :3]  # Extract positions
            else:
                sample['full_positions'] = sample.pop('positions')  # Rename positions
                
            if 'rotations' not in sample:
                sample['full_rotations'] = sample['full_poses'][:, 3:]  # Extract 6D rotations
            else:
                sample['full_rotations'] = sample.pop('rotations')  # Rename rotations
                
        elif 'positions' in sample and 'attention_mask' in sample:
            # Legacy case - only positions available
            sample['full_positions'] = sample.pop('positions')
            sample['full_attention_mask'] = sample.pop('attention_mask')
            
            # Create empty rotation tensor with zeros (fall back, should not happen with updated dataset)
            sample['full_rotations'] = torch.zeros_like(sample['full_positions']).repeat(1, 2)  # [N, 3] -> [N, 6]
            
            # Create combined poses tensor
            sample['full_poses'] = torch.cat([sample['full_positions'], sample['full_rotations']], dim=1)
            print(f"Warning: Created placeholder rotations for sample without rotation data")
        else:
            print(f"Warning: Could not find 'poses' or 'positions' in sample from {sequence_path}")
            # Assign placeholder tensors if data is missing to ensure consistent keys
            dummy_pos = torch.zeros((self.trajectory_length, 3), dtype=torch.float)
            dummy_rot = torch.zeros((self.trajectory_length, 6), dtype=torch.float)
            dummy_poses = torch.zeros((self.trajectory_length, 9), dtype=torch.float)
            dummy_mask = torch.zeros(self.trajectory_length, dtype=torch.float)
            
            sample['full_positions'] = dummy_pos
            sample['full_rotations'] = dummy_rot
            sample['full_poses'] = dummy_poses
            sample['full_attention_mask'] = dummy_mask
        # -------------------------------------
        
        # Add segment_idx if available in original metadata
        if 'metadata' in original_traj_item and 'segment_idx' in original_traj_item['metadata']:
             sample['segment_idx'] = torch.tensor(original_traj_item['metadata']['segment_idx'], dtype=torch.long)
        else:
             # Assign a default or handle missing segment_idx appropriately
             sample['segment_idx'] = torch.tensor(-1, dtype=torch.long) # Use -1 to indicate missing
             
        # Ensure necessary data types for collation (e.g., object_id to tensor)
        # Use .get with default to handle potentially missing keys from base dataset
        sample['object_id'] = torch.tensor(sample.get('object_id', -1), dtype=torch.long)
        
        # Convert first_position if it exists and is numpy
        first_pos = sample.get('first_position')
        if isinstance(first_pos, np.ndarray):
             sample['first_position'] = torch.from_numpy(first_pos).float()
        elif first_pos is None and sample.get('use_displacements', False):
             # Assign a default if using displacements and it's missing
             sample['first_position'] = torch.zeros(3, dtype=torch.float)
        # Ensure it's a tensor or None if not needed/present
        elif not isinstance(first_pos, torch.Tensor) and first_pos is not None:
             print(f"Warning: Unexpected type for first_position: {type(first_pos)}. Setting to None.")
             sample['first_position'] = None # Or handle error appropriately

        # Make sure trajectory-specific pointcloud is included if available
        if 'trajectory_specific_pointcloud' not in sample:
            try:
                # Get trajectory-specific point cloud directly from the original dataset
                traj_pc = self.individual_datasets[dataset_idx].get_trajectory_specific_pointcloud(local_idx)
                if traj_pc is not None:
                    sample['trajectory_specific_pointcloud'] = traj_pc
            except Exception as e:
                print(f"Warning: Could not get trajectory-specific point cloud for sample {idx}: {e}")

        # Add complete scene pointcloud for evaluation visualization
        if 'scene_pointcloud' not in sample:
            try:
                # Get complete scene point cloud from the dataset
                scene_pc = self.individual_datasets[dataset_idx].get_scene_pointcloud()
                if scene_pc is not None and len(scene_pc) > 0:
                    # Convert to tensor if it's numpy array
                    if isinstance(scene_pc, np.ndarray):
                        scene_pc = torch.from_numpy(scene_pc).float()
                    sample['scene_pointcloud'] = scene_pc
                    # Also store the sequence info for debugging
                    sample['scene_pointcloud_source'] = f"sequence_{dataset_idx}"
                else:
                    print(f"Warning: No scene point cloud available for sequence {dataset_idx}")
            except Exception as e:
                print(f"Warning: Could not get scene point cloud for sample {idx}: {e}")

        return sample
    