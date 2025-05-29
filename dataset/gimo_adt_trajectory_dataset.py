#!/usr/bin/env python3
# Dataset class for loading Aria Digital Twin trajectories

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import traceback
from typing import Dict, List, Tuple, Optional, Union
import sys
import pickle
import hashlib
import time
from scipy.spatial import cKDTree  # Add import for efficient nearest neighbor search
from scipy.spatial.transform import Rotation as R

# Import geometry utilities
from utils.geometry_utils import rotation_matrix_to_6d, rotation_6d_to_matrix

class GimoAriaDigitalTwinTrajectoryDataset(Dataset):
    """Dataset for loading trajectories from Aria Digital Twin sequences."""

    def __init__(
        self,
        sequence_path: str,
        config=None,  # Add config parameter
        trajectory_length: int = 100,
        skip_frames: int = 5,
        max_objects: Optional[int] = None,
        device_num: int = 0,
        transform=None,
        load_pointcloud: bool = True,
        pointcloud_subsample: int = 1,  # Default changed to 1 to use full resolution
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        min_motion_threshold: float = 0.0,  # Minimum motion threshold (in meters)
        min_motion_percentile: float = 0.0,  # Filter trajectories below this percentile of motion
        use_displacements: bool = False,  # Whether to use displacements instead of absolute positions
        detect_motion_segments: bool = True,  # Whether to detect and extract active motion segments
        motion_velocity_threshold: float = 0.05,  # Threshold in m/s for detecting active motion
        min_segment_frames: int = 5,  # Minimum number of frames for a valid motion segment
        max_stationary_frames: int = 3,   # Maximum consecutive stationary frames allowed in a motion segment
        trajectory_pointcloud_radius: float = 0.5,  # Radius around trajectory to collect points (meters)
        force_use_cache: bool = False,    # Force use any compatible cache file, ignoring parameter mismatches
    ):
        """
        Initialize the ADT trajectory dataset.

        Args:
            sequence_path: Path to the ADT sequence
            config: Optional configuration object with dataset parameters
            trajectory_length: Length of trajectories to extract (if config not provided)
            skip_frames: Number of frames to skip between samples (if config not provided)
            max_objects: Maximum number of objects to include (None for all)
            device_num: Device number for ADT data provider
            transform: Optional transform to apply to trajectories
            load_pointcloud: Whether to load the MPS pointcloud (if config not provided)
            pointcloud_subsample: Subsample factor for pointcloud (if config not provided)
            use_cache: Whether to use caching for trajectories (if config not provided)
            cache_dir: Directory to store cached data (if config not provided)
            min_motion_threshold: Minimum motion threshold in meters (if config not provided)
            min_motion_percentile: Filter trajectories below this percentile of motion (if config not provided)
            use_displacements: Whether to use displacements instead of absolute positions (if config not provided)
            detect_motion_segments: Whether to detect and extract active motion segments (if config not provided)
            motion_velocity_threshold: Threshold in m/s for detecting active motion (if config not provided)
            min_segment_frames: Minimum number of frames for a valid motion segment (if config not provided)
            max_stationary_frames: Maximum consecutive stationary frames allowed in a motion segment (if config not provided)
            trajectory_pointcloud_radius: Radius around trajectory to collect points (meters)
            force_use_cache: Force use any compatible cache file, ignoring parameter mismatches
        """
        from projectaria_tools.projects.adt import (
            AriaDigitalTwinDataProvider,
            AriaDigitalTwinDataPathsProvider,
            MotionType
        )

        self.sequence_path = sequence_path
        
        # Use config values if provided, otherwise use the explicitly passed parameters
        if config is not None:
            self.trajectory_length = getattr(config, 'trajectory_length', trajectory_length)
            self.skip_frames = getattr(config, 'skip_frames', skip_frames)
            self.load_pointcloud = getattr(config, 'load_pointcloud', load_pointcloud)
            self.pointcloud_subsample = getattr(config, 'pointcloud_subsample', pointcloud_subsample)
            self.use_cache = getattr(config, 'use_cache', use_cache)
            self.min_motion_threshold = getattr(config, 'min_motion_threshold', min_motion_threshold)
            self.min_motion_percentile = getattr(config, 'min_motion_percentile', min_motion_percentile)
            self.use_displacements = getattr(config, 'use_displacements', use_displacements)
            self.detect_motion_segments = getattr(config, 'detect_motion_segments', detect_motion_segments)
            self.motion_velocity_threshold = getattr(config, 'motion_velocity_threshold', motion_velocity_threshold)
            self.min_segment_frames = getattr(config, 'min_segment_frames', min_segment_frames)
            self.max_stationary_frames = getattr(config, 'max_stationary_frames', max_stationary_frames)
            self.trajectory_pointcloud_radius = getattr(config, 'trajectory_pointcloud_radius', trajectory_pointcloud_radius)
            self.force_use_cache = getattr(config, 'force_use_cache', force_use_cache) # Get force_use_cache from config
            # For cache_dir, don't use config directly - it might be set based on save_path
        else:
            self.trajectory_length = trajectory_length
            self.skip_frames = skip_frames
            self.load_pointcloud = load_pointcloud
            self.pointcloud_subsample = pointcloud_subsample
            self.use_cache = use_cache
            self.min_motion_threshold = min_motion_threshold
            self.min_motion_percentile = min_motion_percentile
            self.use_displacements = use_displacements
            self.detect_motion_segments = detect_motion_segments
            self.motion_velocity_threshold = motion_velocity_threshold
            self.min_segment_frames = min_segment_frames
            self.max_stationary_frames = max_stationary_frames
            self.trajectory_pointcloud_radius = trajectory_pointcloud_radius
            self.force_use_cache = force_use_cache # Use passed parameter if no config
        

        self.load_pointcloud = self.load_pointcloud

        # These parameters don't typically come from config
        self.max_objects = max_objects
        self.transform = transform
        self.pointcloud = None
        self.trajectories = []
        
        # Cache directory is now determined by the caller (GIMOMultiSequenceDataset) or defaults
        self.cache_dir = cache_dir if cache_dir is not None else './trajectory_cache'

        # Create cache directory if it doesn't exist and caching is enabled
        if self.use_cache and not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir)
                print(f"Created cache directory: {self.cache_dir}")
            except Exception as e:
                print(f"Warning: Could not create cache directory: {e}")
                self.use_cache = False

        # --- Log cache directory being used by this specific dataset instance ---
        print(f"GimoAriaDigitalTwinTrajectoryDataset using cache directory: {self.cache_dir}")
        # -----------------------------------------------------------------------

        print(f"Loading ADT sequence from: {sequence_path}")
        
        # Check for cached data first
        if self.use_cache and self._load_from_cache():
            print(f"Loaded cached trajectories for {sequence_path}")
            return
            
        # Create data provider
        try:
            paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
            
            # Use the correct method name: get_datapaths instead of get_data_paths
            data_paths = paths_provider.get_datapaths(False)
            if data_paths is None:
                raise ValueError(f"Failed to get data paths from {sequence_path}")
                
            self.adt_provider = AriaDigitalTwinDataProvider(data_paths)
            
            # Load pointcloud if requested
            if self.load_pointcloud:
                self._load_pointcloud()
            
            # Extract trajectories
            start_time = time.time()
            self._extract_trajectories()
            print(f"Trajectory extraction took {time.time() - start_time:.2f} seconds")
            
            # Generate trajectory-specific pointclouds (new step)
            if self.pointcloud is not None:
                print("Generating trajectory-specific point clouds...")
                self._generate_trajectory_specific_pointclouds()
                print(f"Completed generating trajectory-specific point clouds for {len(self.trajectories)} trajectories")
            
            # Save to cache if caching is enabled
            if self.use_cache:
                self._save_to_cache()
            
        except Exception as e:
            print(f"Error loading ADT sequence: {e}")
            traceback.print_exc()
            raise
    
    def _filter_points_by_trajectory(self, trajectory_positions, radius=None):
        """
        Filter the full scene point cloud to include only points within a specified radius of any trajectory point.
        
        Args:
            trajectory_positions: Numpy array of shape [N, 3] containing trajectory positions
            radius: Radius in meters around each trajectory point to include scene points (defaults to self.trajectory_pointcloud_radius)
            
        Returns:
            Numpy array of shape [M, 3] containing filtered point cloud
        """
        if self.pointcloud is None or len(self.pointcloud) == 0:
            print("Warning: No point cloud available for filtering")
            return np.zeros((0, 3), dtype=np.float32)
            
        if trajectory_positions is None or len(trajectory_positions) == 0:
            print("Warning: Empty trajectory provided for point cloud filtering")
            return np.zeros((0, 3), dtype=np.float32)
            
        # Use instance default radius if not specified
        if radius is None:
            radius = self.trajectory_pointcloud_radius
        
        # Use KDTree for efficient nearest neighbor queries
        try:
            # Convert trajectory positions to numpy if they're tensors
            if isinstance(trajectory_positions, torch.Tensor):
                trajectory_positions = trajectory_positions.detach().cpu().numpy()
                

                
            # Build KD-tree on trajectory points
            trajectory_tree = cKDTree(trajectory_positions)
            
            # Use KD-tree to find all points within radius of any trajectory point
            distances, _ = trajectory_tree.query(self.pointcloud, k=1)
            mask = distances <= radius
            
            # Return filtered points
            filtered_points = self.pointcloud[mask]
            
            # For debugging
            print(f"Filtered point cloud: {len(filtered_points)} points (from {len(self.pointcloud)} total)")
            
            return filtered_points
            
        except Exception as e:
            print(f"Error filtering point cloud by trajectory: {e}")
            traceback.print_exc()
            return np.zeros((0, 3), dtype=np.float32)
    
    def _generate_trajectory_specific_pointclouds(self):
        """
        Generate and store trajectory-specific pointclouds for each trajectory in the dataset.
        These filtered point clouds only include points within trajectory_pointcloud_radius
        of any point in the trajectory.
        """
        if self.pointcloud is None or len(self.pointcloud) == 0:
            print("Warning: No point cloud available to generate trajectory-specific point clouds")
            return
            
        print(f"Generating trajectory-specific point clouds for {len(self.trajectories)} trajectories...")
        
        # Create a KD-tree for the full point cloud once (for efficiency)
        full_pc_tree = cKDTree(self.pointcloud)
        
        for i, traj_item in enumerate(self.trajectories):
            try:
                # Get trajectory poses and extract positions (first 3 dimensions)
                poses_tensor = traj_item['trajectory_data']['poses']
                positions_tensor = poses_tensor[:, :3]  # Extract first 3 dimensions (x, y, z)
                
                # Get attention mask to filter out padding
                mask = traj_item['trajectory_data']['attention_mask']
                
                # Extract only valid positions (where mask is 1)
                if isinstance(mask, torch.Tensor):
                    valid_indices = torch.where(mask > 0.5)[0]
                    valid_positions = positions_tensor[valid_indices]
                else:
                    # Fallback if mask isn't a tensor
                    valid_indices = np.where(np.array(mask) > 0.5)[0]
                    valid_positions = positions_tensor[valid_indices]
                

                
                # Filter the pointcloud
                filtered_pc = self._filter_points_by_trajectory(valid_positions)
                
                # Store the filtered point cloud with the trajectory metadata
                if 'metadata' not in traj_item:
                    traj_item['metadata'] = {}
                
                # Store the filtered point cloud
                traj_item['trajectory_specific_pointcloud'] = filtered_pc
                
                # For progress reporting
                if (i+1) % 10 == 0 or i == 0 or i == len(self.trajectories)-1:
                    print(f"  Processed {i+1}/{len(self.trajectories)} trajectories")
                
            except Exception as e:
                print(f"Error generating trajectory-specific point cloud for trajectory {i}: {e}")
                traceback.print_exc()
                # Set an empty point cloud as fallback
                traj_item['trajectory_specific_pointcloud'] = np.zeros((0, 3), dtype=np.float32)
        
        print(f"Completed generating trajectory-specific point clouds")
    
    def _get_cache_filename(self):
        """Generate a unique filename for the cache based on the dataset parameters."""
        # Get sequence name instead of full path for more portable caches
        sequence_name = os.path.basename(self.sequence_path)
        
        # When force_use_cache is enabled, don't include parameters in the hash
        # This will create a simpler filename that can be matched across different parameter configurations
        if self.force_use_cache:
            # Just use sequence name with a simple file extension
            return os.path.join(self.cache_dir, f"{sequence_name}_cache.pkl")
        
        # Create a unique identifier based on relevant parameters, excluding the full path
        params = {
            'sequence_name': sequence_name,  # Use sequence name, not full path
            'trajectory_length': self.trajectory_length,
            'skip_frames': self.skip_frames,
            'max_objects': self.max_objects,
            'use_displacements': self.use_displacements,
            'detect_motion_segments': self.detect_motion_segments,
            'motion_velocity_threshold': self.motion_velocity_threshold,
            'min_segment_frames': self.min_segment_frames,
            'max_stationary_frames': self.max_stationary_frames,
            'min_motion_threshold': self.min_motion_threshold,
            'min_motion_percentile': self.min_motion_percentile,
            'pointcloud_subsample': self.pointcloud_subsample,
            'trajectory_pointcloud_radius': self.trajectory_pointcloud_radius, # Add new parameter to cache key
            'include_bbox_corners': True # New parameter for bbox corners
        }
        
        # Generate hash from parameters
        params_str = str(params)
        hash_obj = hashlib.md5(params_str.encode())
        hash_str = hash_obj.hexdigest()
        
        # Create filename using sequence name and hash
        return os.path.join(self.cache_dir, f"{sequence_name}_{hash_str}.pkl")
    
    def _save_to_cache(self):
        """Save trajectories and scene features to cache."""
        if not self.use_cache:
            return False
            
        cache_file = self._get_cache_filename()
        
        try:
            # Get sequence name for consistency
            sequence_name = os.path.basename(self.sequence_path)
            
            # Prepare data to save
            cache_data = {
                'trajectories': self.trajectories,
                'pointcloud': self.pointcloud,
                'params': {
                    'sequence_name': sequence_name,  # Store sequence name, not full path
                    'trajectory_length': self.trajectory_length,
                    'skip_frames': self.skip_frames,
                    'max_objects': self.max_objects,
                    'use_displacements': self.use_displacements,
                    'detect_motion_segments': self.detect_motion_segments,
                    'motion_velocity_threshold': self.motion_velocity_threshold,
                    'min_segment_frames': self.min_segment_frames,
                    'max_stationary_frames': self.max_stationary_frames,
                    'min_motion_threshold': self.min_motion_threshold,
                    'min_motion_percentile': self.min_motion_percentile,
                    'pointcloud_subsample': self.pointcloud_subsample,
                    'trajectory_pointcloud_radius': self.trajectory_pointcloud_radius, # Save new parameter
                    'include_bbox_corners': True # Save new parameter state
                }
            }
            
            # Save to file
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            print(f"Saved trajectory cache to {cache_file}")
            return True
        except Exception as e:
            print(f"Error saving trajectory cache: {e}")
            return False
    
    def _load_from_cache(self):
        """Load trajectories and scene features from cache if available."""
        if not self.use_cache:
            return False
            
        # Get current cache file path with hash
        cache_file = self._get_cache_filename()
        
        # Get sequence name for potential fallback search
        sequence_name = os.path.basename(self.sequence_path)
        
        # Try loading the exact cache file first
        if not os.path.exists(cache_file):
            print(f"No cache file found at {cache_file}")
            
            # Fallback: try to find any cache file for this sequence
            potential_files = []
            if os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.startswith(f"{sequence_name}_") and filename.endswith(".pkl"):
                        potential_files.append(os.path.join(self.cache_dir, filename))
            
            if potential_files:
                # Sort by modification time (newest first)
                potential_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                cache_file = potential_files[0]
                print(f"Found potential cache file: {cache_file}")
            else:
                print(f"No potential cache files found for {sequence_name}")
                return False
            
        try:
            # Load data from file
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Get sequence name for comparison (not full path)
            sequence_name = os.path.basename(self.sequence_path)
            
            # Print the actual parameters for debugging
            stored_params = cache_data['params']
            current_params = {
                'sequence_name': sequence_name,
                'trajectory_length': self.trajectory_length,
                'skip_frames': self.skip_frames,
                'max_objects': self.max_objects,
                'use_displacements': self.use_displacements,
                'detect_motion_segments': self.detect_motion_segments,
                'motion_velocity_threshold': self.motion_velocity_threshold,
                'min_segment_frames': self.min_segment_frames,
                'max_stationary_frames': self.max_stationary_frames,
                'min_motion_threshold': self.min_motion_threshold,
                'min_motion_percentile': self.min_motion_percentile,
                'pointcloud_subsample': self.pointcloud_subsample,
                'trajectory_pointcloud_radius': self.trajectory_pointcloud_radius,
                'include_bbox_corners': True # Current setting for bbox corners
            }
            
            # Verify parameters match, with extra debug info
            stored_sequence = stored_params.get('sequence_name', os.path.basename(stored_params.get('sequence_path', '')))
            if stored_sequence != sequence_name:
                print(f"Sequence mismatch: {stored_sequence} vs {sequence_name}")
                return False
                
            # Check other critical parameters (skip if force_use_cache is enabled)
            if not self.force_use_cache:
                critical_params = ['trajectory_length', 'skip_frames', 'use_displacements', 
                                   'motion_velocity_threshold',
                                'include_bbox_corners'] # Add new param to critical check
                
                for param in critical_params:
                    stored_value = stored_params.get(param, None)
                    current_value = current_params.get(param, None)
                    if stored_value != current_value:
                        print(f"Parameter mismatch: {param} - stored: {stored_value}, current: {current_value}")
                        return False
            else:
                print("Force use cache enabled: Ignoring parameter mismatches")
            
            # Load data
            self.trajectories = cache_data['trajectories']
            self.pointcloud = cache_data['pointcloud']

            # Bbox corners are already part of self.trajectories, no separate loading needed here
            # if they were saved correctly. The check for 'include_bbox_corners' ensures compatibility.
            
            print(f"Successfully loaded {len(self.trajectories)} trajectories from cache: {cache_file}")
            return True
        except Exception as e:
            print(f"Error loading trajectory cache: {e}")
            return False
    
    def _load_pointcloud(self):
        """Load the MPS pointcloud from the sequence."""
        if not os.path.exists(self.sequence_path):
            print(f"Error: Sequence path {self.sequence_path} does not exist")
            self._set_default_scene_bounds() # Set default bounds if path invalid
            return

        try:
            # Import projectaria_tool
            import projectaria_tools.core.mps as aria_mps
            from projectaria_tools.core.mps.utils import filter_points_from_confidence
            
            print(f"Loading MPS point cloud from {self.sequence_path}...")
            
            # Create paths provider and get data paths
            mps_path = os.path.join(self.sequence_path, "mps")
            if not os.path.exists(mps_path):
                print(f"MPS directory not found at {mps_path}")
                self._set_default_scene_bounds() # Set default bounds if no MPS dir
                return
                
            paths_provider = aria_mps.MpsDataPathsProvider(mps_path)
            data_paths = paths_provider.get_data_paths()
            if not data_paths:
                print("Failed to get MPS data paths")
                self._set_default_scene_bounds() # Set default bounds
                return
                
            # Create MPS data provider
            mps_provider = aria_mps.MpsDataProvider(data_paths)
            
            # Check if semidense point cloud is available
            if not mps_provider.has_semidense_point_cloud():
                print("No semidense point cloud available")
                self._set_default_scene_bounds() # Set default bounds
                return
                
            # Get semidense point cloud
            point_cloud = mps_provider.get_semidense_point_cloud()
            if not point_cloud:
                print("Failed to get semidense point cloud")
                self._set_default_scene_bounds() # Set default bounds
                return
                
            print(f"Loaded {len(point_cloud)} semidense points")

            # Filter the point cloud using thresholds
            inverse_distance_std_threshold = 0.001
            distance_std_threshold = 0.15
            point_cloud = filter_points_from_confidence(
                point_cloud, 
                inverse_distance_std_threshold,
                distance_std_threshold
            )
            
            # Convert to numpy array
            points = np.array([point.position_world for point in point_cloud])
            
            # Subsample points if needed
            if self.pointcloud_subsample > 1 and len(points) > 0:
                indices = np.random.choice(
                    len(points), 
                    size=len(points) // self.pointcloud_subsample, 
                    replace=False
                )
                points = points[indices]
                print(f"Subsampled to {len(points)} points (1/{self.pointcloud_subsample} of original)")
            
            # Store the pointcloud
            self.pointcloud = points
            print(f"Pointcloud loaded with {len(self.pointcloud)} points")
        except Exception as e:
            print(f"Error loading MPS point cloud: {e}")
            traceback.print_exc()
            
    def _calculate_trajectory_motion(self, positions, timestamps=None):
        """
        Calculate the total motion/displacement of a trajectory.
        
        Args:
            positions: List of position vectors [x,y,z] (extracted from poses if necessary)
            timestamps: Optional list of timestamps corresponding to positions
            
        Returns:
            total_path_length: Sum of distances between consecutive points (absolute distance moved)
            direct_displacement: Direct displacement between first and last points
            max_displacement: Maximum displacement between any point and the start point
            avg_velocity: Average velocity in m/s (with timestamps) or per-frame (without)
        """
        if len(positions) < 2:
            return 0.0, 0.0, 0.0, 0.0  # Return all four values: total_path_length, direct_displacement, max_displacement, avg_velocity
        
        # Convert positions to numpy array if needed
        if isinstance(positions, torch.Tensor):
            positions_np = positions.detach().cpu().numpy()
        elif isinstance(positions, list):
            # If positions is a list of position vectors, convert to numpy array
            # Ensure each position is only [x,y,z] if extracted from [x,y,z,roll,pitch,yaw]
            if len(positions[0]) > 3:
                positions_np = np.array([pos[:3] for pos in positions])
            else:
                positions_np = np.array(positions)
        else:
            positions_np = positions
            
        # Calculate total path length (sum of all point-to-point movements)
        # This represents the absolute distance moved along the trajectory
        diffs = np.diff(positions_np, axis=0)
        segment_distances = np.linalg.norm(diffs, axis=1)
        total_path_length = np.sum(segment_distances)
        
        # Calculate direct start-to-end displacement
        # This is just the straight-line distance between first and last points
        if len(positions_np) > 1:
            direct_displacement = np.linalg.norm(positions_np[-1] - positions_np[0])
        else:
            direct_displacement = 0.0
            
        # Calculate maximum displacement from start
        distances_from_start = np.linalg.norm(positions_np - positions_np[0], axis=1)
        max_displacement = np.max(distances_from_start)
        
        # Calculate velocity using timestamps if available
        if timestamps is not None and len(timestamps) == len(positions_np):
            # Calculate full trajectory duration in seconds
            trajectory_duration_ns = timestamps[-1] - timestamps[0]
            trajectory_duration_s = trajectory_duration_ns / 1e9
            
            if trajectory_duration_s > 0:
                # Average velocity over the entire trajectory (m/s)
                avg_velocity = total_path_length / trajectory_duration_s
                
                # Additional metrics using timestamps
                # Calculate instantaneous velocities
                instant_velocities = []
                for i in range(1, len(positions_np)):
                    dt_ns = timestamps[i] - timestamps[i-1]
                    dt_s = dt_ns / 1e9
                    if dt_s > 0:
                        vel = segment_distances[i-1] / dt_s
                        instant_velocities.append(vel)
                
                if instant_velocities:
                    # These are useful metrics but we'll just store avg_velocity for now
                    max_velocity = np.max(instant_velocities)
                    min_velocity = np.min(instant_velocities)
                    median_velocity = np.median(instant_velocities)
            else:
                avg_velocity = 0.0
        else:
            # Fall back to frame-based velocity if timestamps not available
            avg_velocity = total_path_length / (len(positions_np) - 1) if len(positions_np) > 1 else 0.0
        
        return total_path_length, direct_displacement, max_displacement, avg_velocity

    def _extract_trajectories(self):
        """Extract trajectories from the ADT sequence."""
        # Get timestamps from a reliable stream (usually RGB camera)
        from projectaria_tools.core.stream_id import StreamId
        from projectaria_tools.projects.adt import MotionType
        
        # Use the RGB camera stream ID directly
        rgb_stream_id = StreamId("214-1")  # This is the standard RGB camera ID in projectaria_tools
        
        # Get all timestamps
        all_timestamps = self.adt_provider.get_aria_device_capture_timestamps_ns(rgb_stream_id)
        if not all_timestamps:
            raise ValueError("No timestamps found in dataset")
            
        # Get valid time range
        start_time = self.adt_provider.get_start_time_ns()
        end_time = self.adt_provider.get_end_time_ns()
        
        # Filter timestamps to valid range
        timestamps = [ts for ts in all_timestamps if start_time <= ts <= end_time]
        
        print(f"Found {len(all_timestamps)} total timestamps, {len(timestamps)} within valid range")
        
        if not timestamps:
            raise ValueError("No valid timestamps found in valid time range")
        
        # Get all dynamic objects
        dynamic_objects = []
        all_objects = []
        motion_types_found = set()
        
        # Get all instance IDs
        instance_ids = self.adt_provider.get_instance_ids()
        print(f"Found {len(instance_ids)} total instances in sequence")
        
        # Check each instance to find dynamic objects
        for instance_id in instance_ids:
            try:
                instance_info = self.adt_provider.get_instance_info_by_id(instance_id)
                
                # Track what motion types we found for debugging
                motion_types_found.add(str(instance_info.motion_type))
                
                # Store all objects for fallback
                all_objects.append((instance_id, instance_info))
                
                # Check if this is a dynamic object using the enum
                if instance_info.motion_type == MotionType.DYNAMIC:
                    dynamic_objects.append((instance_id, instance_info))
            except Exception as e:
                print(f"Error processing instance {instance_id}: {e}")
                continue
                
        # print(f"Motion types found in sequence: {motion_types_found}")
        
        # If no dynamic objects found, fall back to using all objects
        if not dynamic_objects:
            print("No dynamic objects found in sequence, using all objects as fallback")
            dynamic_objects = all_objects
            
        if not dynamic_objects:
            raise ValueError("No objects found in sequence")
            
        print(f"Found {len(dynamic_objects)} objects to track in sequence")
        
        # Limit number of objects if specified
        if self.max_objects is not None and len(dynamic_objects) > self.max_objects:
            dynamic_objects = dynamic_objects[:self.max_objects]
            print(f"Using {len(dynamic_objects)} objects (limited by max_objects)")
        
        # Lists to store trajectory motion statistics for filtering
        trajectories_with_motion = []
        
        # Track each object
        num_total_objects_to_track = len(dynamic_objects)
        for i, (obj_id, instance_info) in enumerate(dynamic_objects):
            try:
                # Print progress
                # print(f"  Processing object {i+1}/{num_total_objects_to_track}: ID={obj_id}, Name={instance_info.name}")
                
                # Track object motion - returns a list of segments
                segments = self._track_object_motion(obj_id, timestamps)
                
                # Process each segment as a separate trajectory
                for segment_idx, (poses, tracked_timestamps, bbox_corners_sequence) in enumerate(segments):
                    if len(poses) < 2:
                        # Skip segments that are too short
                        continue
                        
                    # Process this segment
                    self._process_trajectory_segment(
                        poses, 
                        tracked_timestamps, 
                        instance_info, 
                        obj_id, 
                        trajectories_with_motion,
                        segment_idx=segment_idx,
                        bbox_corners_sequence=bbox_corners_sequence
                    )
                
            except Exception as e:
                print(f"Error processing object {obj_id}: {e}")
                traceback.print_exc()
                
        # Apply motion filtering if threshold is set
        if self.min_motion_threshold > 0 or self.min_motion_percentile > 0:
            # Sort trajectories by motion value
            sorted_trajectories = sorted(trajectories_with_motion, key=lambda x: x['motion_value'])
            
            # Apply absolute motion threshold
            if self.min_motion_threshold > 0:
                filtered_trajectories = [t for t in trajectories_with_motion if t['motion_value'] >= self.min_motion_threshold]
                if len(filtered_trajectories) == 0 and len(trajectories_with_motion) > 0:
                    # If filtering removed all trajectories, keep the most moving one
                    filtered_trajectories = [sorted_trajectories[-1]]
                    print(f"Warning: Motion threshold {self.min_motion_threshold}m filtered all trajectories. Keeping only the most moving one.")
                
                num_filtered = len(trajectories_with_motion) - len(filtered_trajectories)
                if num_filtered > 0:
                    print(f"Filtered out {num_filtered} trajectories with total path length < {self.min_motion_threshold}m")
                
                trajectories_with_motion = filtered_trajectories
            
            # Apply percentile-based filtering
            if self.min_motion_percentile > 0 and len(sorted_trajectories) > 0:
                cutoff_idx = int(len(sorted_trajectories) * (self.min_motion_percentile / 100.0))
                filtered_trajectories = sorted_trajectories[cutoff_idx:]
                
                if len(filtered_trajectories) == 0:
                    # Keep at least one trajectory
                    filtered_trajectories = [sorted_trajectories[-1]]
                    print(f"Warning: Percentile {self.min_motion_percentile}% filtered all trajectories. Keeping only the most moving one.")
                
                num_filtered = len(sorted_trajectories) - len(filtered_trajectories)
                if num_filtered > 0:
                    min_motion = filtered_trajectories[0]['motion_value']
                    print(f"Filtered out {num_filtered} trajectories below {self.min_motion_percentile}% motion percentile (min total path length: {min_motion:.3f}m)")
                
                trajectories_with_motion = filtered_trajectories
            
            # Sort trajectories by motion in descending order
            trajectories_with_motion.sort(key=lambda x: x['motion_value'], reverse=True)
            
            # print detail of kept trajectories: object name, segment index, duration
            # for t in trajectories_with_motion:
            #     print(f"Object: {t['metadata']['name']}, Segment Index: {t['metadata']['segment_idx']}, Duration: {t['metadata']['segment_duration_s']:.2f}s")
            
            # Print motion statistics of kept trajectories
            if trajectories_with_motion:
                motion_values = [t['motion_value'] for t in trajectories_with_motion]
                print(f"Keeping {len(trajectories_with_motion)} trajectories with motion metrics:")
                print(f"  Min total path length: {min(motion_values):.3f}m")
                print(f"  Max total path length: {max(motion_values):.3f}m")
                print(f"  Avg total path length: {sum(motion_values)/len(motion_values):.3f}m")
        
        # Store final trajectories (remove the motion_value field)
        self.trajectories = [
            {
                'trajectory_data': t['trajectory_data'],
                'metadata': t['metadata']
            }
            for t in trajectories_with_motion
        ]
        
        print(f"Final dataset contains {len(self.trajectories)} trajectories")
    
    def _process_trajectory_segment(self, poses, tracked_timestamps, instance_info, obj_id, trajectories_with_motion, segment_idx=None, bbox_corners_sequence=None):
        """
        Process a single trajectory segment and add it to the trajectories list.
        
        Args:
            poses: List of pose vectors [x, y, z, roll, pitch, yaw]
            tracked_timestamps: List of timestamps for each pose
            instance_info: Object instance information
            obj_id: Object ID
            trajectories_with_motion: List to store processed trajectories
            segment_idx: Optional index of the segment (for multi-segment objects)
            bbox_corners_sequence: List of 8 corners of the Oriented Bounding Box (OBB) for each timestamp
        """
        # Extract positions (first 3 elements of each pose) for motion metrics
        positions = [pose[:3] for pose in poses]
        

        total_path_length, direct_displacement, max_displacement, avg_velocity = self._calculate_trajectory_motion(positions, tracked_timestamps)
        
        # Convert to numpy array and ensure consistent 2D shape [N, 9]
        poses_array = np.array(poses)
        if poses_array.ndim > 2:
            poses_array = poses_array.reshape(-1, 9)  # Reshape to [N, 9] for [x,y,z,r6d_1,r6d_2,r6d_3,r6d_4,r6d_5,r6d_6]
        
        poses_tensor = torch.tensor(poses_array, dtype=torch.float32)
        real_length = min(len(poses_tensor), self.trajectory_length)
        
        if self.use_displacements and real_length > 1:
            # Create a new tensor to store the modified poses with displacements for positions
            modified_tensor = torch.zeros_like(poses_tensor)
            
            # Keep the first pose as absolute coordinates and rotations
            modified_tensor[0] = poses_tensor[0]
            
            # Calculate displacements for positions (first 3 elements) after the first
            modified_tensor[1:real_length, :3] = poses_tensor[1:real_length, :3] - poses_tensor[:real_length-1, :3]
            
            # Keep absolute rotations (last 6 elements) for all poses
            modified_tensor[1:real_length, 3:] = poses_tensor[1:real_length, 3:]
            
            # Replace poses with this new representation
            poses_tensor = modified_tensor
            
            # No need to store first_position separately in metadata
            metadata_first_position = None
        else:
            metadata_first_position = None
        
        # Create attention mask (ones for real data, zeros for padding)
        attention_mask = torch.zeros(self.trajectory_length, dtype=torch.float32)
        attention_mask[:real_length] = 1.0
        
        # Pad the poses tensor if needed
        if len(poses_tensor) < self.trajectory_length:
            # Make sure padding has the same shape [M, 9]
            padding = torch.zeros(self.trajectory_length - len(poses_tensor), 9, dtype=torch.float32)
            poses_tensor = torch.cat([poses_tensor, padding], dim=0)
        else:
            poses_tensor = poses_tensor[:self.trajectory_length]
        
        # Create trajectory data dict
        trajectory_data = {
            'poses': poses_tensor,  # Now contains [x, y, z, r6d_1, r6d_2, r6d_3, r6d_4, r6d_5, r6d_6]
            'attention_mask': attention_mask,
        }
        
        # Add bbox_corners_sequence to trajectory_data after processing
        if bbox_corners_sequence is not None and len(bbox_corners_sequence) > 0:
            # Convert list of [8,3] arrays to a single numpy array [N, 8, 3]
            bbox_corners_array = np.array(bbox_corners_sequence, dtype=np.float32)

            bbox_corners_tensor = torch.tensor(bbox_corners_array, dtype=torch.float32)
            
            # Pad or truncate bbox_corners_tensor to self.trajectory_length
            # Real length is determined by the poses/attention_mask
            current_len_bbox = bbox_corners_tensor.shape[0]
            if current_len_bbox < self.trajectory_length:
                padding_bbox = torch.zeros(self.trajectory_length - current_len_bbox, 8, 3, dtype=torch.float32)
                padded_bbox_corners_tensor = torch.cat([bbox_corners_tensor, padding_bbox], dim=0)
            else:
                padded_bbox_corners_tensor = bbox_corners_tensor[:self.trajectory_length, :, :]
            
            trajectory_data['bbox_corners'] = padded_bbox_corners_tensor
        else:
            # If no bbox data, add zeros as placeholder
            trajectory_data['bbox_corners'] = torch.zeros(self.trajectory_length, 8, 3, dtype=torch.float32)

        # Create metadata with only essential information
        metadata = {
            'name': instance_info.name,
            'prototype_name': instance_info.name.split('_')[0] if '_' in instance_info.name else 'unknown',
            'id': int(obj_id),
            'sequence': self.sequence_path.split('/')[-1],
            'total_path_length': float(total_path_length), # Motion metrics always in original scale
            'direct_displacement': float(direct_displacement),
            'max_displacement': float(max_displacement),
            'avg_velocity': float(avg_velocity),
            'is_displacement': self.use_displacements,
            'active_motion_segment': True,  # Flag indicating this is an active motion segment
            'segment_duration_s': float((tracked_timestamps[-1] - tracked_timestamps[0]) / 1e9) if len(tracked_timestamps) > 1 else 0.0,
            'original_length': len(poses),  # Store the length of the active segment
            'start_timestamp_ns': int(tracked_timestamps[0]) if tracked_timestamps else None,  # Store segment start timestamp

        }
        
        # Add segment index if provided
        if segment_idx is not None:
            metadata['segment_idx'] = segment_idx
        
        # Extract category and subcategory if available
        metadata['category'] = getattr(instance_info, 'category', 'unknown')
        metadata['subcategory'] = getattr(instance_info, 'subcategory', 'unknown')
        
        # Extract or assign category_id directly
        # First try to get a direct category_id/uid from instance_info
        category_id = getattr(instance_info, 'category_id', None)
        if category_id is None:
            # If not available, use a simple hash of the category string as backup
            # This provides a consistent ID for the same category string
            category_string = metadata['category']
            category_id = hash(category_string) % 10000  # Limit to a reasonable range
        
        # Store the category_id in metadata
        metadata['category_id'] = int(category_id)
        
        # Create complete trajectory item
        trajectory_item = {
            'trajectory_data': trajectory_data,
            'metadata': metadata,
            'motion_value': total_path_length  # Store total path length for filtering (absolute distance moved)
        }
        
        # Add bbox_corners_sequence to metadata
        if bbox_corners_sequence is not None:
            metadata['bbox_corners_sequence'] = bbox_corners_sequence
        
        trajectories_with_motion.append(trajectory_item)
    
    def _track_object_motion(self, object_id, timestamps):
        """
        Track an object's motion across a sequence.
        
        Args:
            object_id: ID of the object to track
            timestamps: List of timestamps to track at
            
        Returns:
            list: List of (poses, timestamps, bbox_corners_sequence) tuples for all active motion segments,
                  where poses contains both position and orientation [x, y, z, roll, pitch, yaw]
                  or a list with a single tuple of the entire trajectory if motion detection is disabled
        """
        from collections import deque
        import numpy as np
        
        # Lists to store trajectory data
        poses = []  # Will store [x, y, z, roll, pitch, yaw]
        tracked_timestamps = []
        bbox_corners_sequence = [] # New list to store 8 corners of OBB for each timestamp
        
        # Track object at each timestamp
        time_idx = 0  # For debug print limiting
        
        try:
            # Use skip_frames parameter to skip frames
            for i in range(0, len(timestamps), self.skip_frames):
                ts = timestamps[i]
                time_idx += 1
                try:
                    # Get 3D bounding box for the dynamic object
                    # Get all bounding boxes at this timestamp
                    bbox3d_with_dt = self.adt_provider.get_object_3d_boundingboxes_by_timestamp_ns(ts)
                    
                    if not bbox3d_with_dt.is_valid():
                        # Skip if we don't have valid bounding boxes at this timestamp
                        continue
                        
                    # Get the bounding box data
                    bboxes3d = bbox3d_with_dt.data()
                    
                    # Check if this object exists in the bounding boxes
                    if object_id not in bboxes3d:
                        # Skip this timestamp if object doesn't have a bounding box
                        continue
                    
                    # Get the specific bounding box for this object
                    bbox = bboxes3d[object_id]
                    
                    # Extract the center position from the transformation
                    center = bbox.transform_scene_object.translation()[0]
                    pos = [center[0], center[1], center[2]]
                    
                    # Extract rotation matrix from transform_scene_object
                    rotation_matrix = bbox.transform_scene_object.rotation().to_matrix()
                    
                    # Convert rotation matrix to 6D representation instead of Euler angles
                    rotation_6d = rotation_matrix_to_6d(rotation_matrix)
                    
                    # Create combined pose [x, y, z, r6d_1, r6d_2, r6d_3, r6d_4, r6d_5, r6d_6]
                    pose = pos + rotation_6d.tolist()
                    
                    # Store the combined pose and timestamp
                    poses.append(pose)
                    tracked_timestamps.append(ts)

                    # --- Extract 8 corners of the Oriented Bounding Box (OBB) ---
                    # sx = bbox.extent_x # Causes AttributeError
                    # sy = bbox.extent_y # Causes AttributeError
                    # sz = bbox.extent_z # Causes AttributeError

                    # Use dimensions from AABB as a fallback, acknowledging this is not the true OBB extent if rotated
                    aabb = bbox.aabb
                    sx = aabb[1] - aabb[0]  # width from world AABB
                    sy = aabb[3] - aabb[2]  # height from world AABB
                    sz = aabb[5] - aabb[4]  # depth from world AABB


                    local_corners = np.array([
                        [-sx/2, -sy/2, -sz/2], [sx/2, -sy/2, -sz/2], [sx/2, sy/2, -sz/2], [-sx/2, sy/2, -sz/2],
                        [-sx/2, -sy/2, sz/2], [sx/2, -sy/2, sz/2], [sx/2, sy/2, sz/2], [-sx/2, sy/2, sz/2]
                    ])

                    # Get the 4x4 transformation matrix - FIX: SE3 may not have matrix() method
                    try:
                        # Try the original method first
                        T_scene_object = bbox.transform_scene_object.matrix()
                    except AttributeError:
                        # If matrix() not available, construct the matrix manually
                        rotation = bbox.transform_scene_object.rotation().to_matrix()
                        translation = bbox.transform_scene_object.translation()[0]
                        
                        # Construct 4x4 transformation matrix
                        T_scene_object = np.eye(4)
                        T_scene_object[:3, :3] = rotation
                        T_scene_object[:3, 3] = translation
                        

                    # Transform local corners to world coordinates
                    local_corners_h = np.hstack((local_corners, np.ones((8, 1)))) # Homogeneous coordinates
                    world_corners_h = (T_scene_object @ local_corners_h.T).T
                    world_corners = world_corners_h[:, :3] # De-homogenize

                    bbox_corners_sequence.append(world_corners)
                    # --------------------------------------------------------------

                except Exception as e:
                    if time_idx % 100 == 0 or time_idx == 1:
                        print(f"Warning: Could not get pose for object {object_id} at time {ts}: {e}")
                    # Skip this timestamp
                    continue
        except Exception as e:
            print(f"Error tracking object {object_id}: {e}")
            traceback.print_exc()
        
        # If we didn't collect enough points or motion detection is disabled, return raw trajectory as a single segment
        if len(poses) < self.min_segment_frames or not self.detect_motion_segments:
            return [(poses, tracked_timestamps, bbox_corners_sequence)]  # Return as a list with one segment
        
        # Calculate velocity between each pair of points - only considering position component (not rotation)
        velocities = []
        time_differences = []
        
        for i in range(1, len(poses)):
            # Calculate time difference in seconds
            dt = (tracked_timestamps[i] - tracked_timestamps[i-1]) / 1e9  # Convert ns to seconds
            
            if dt <= 0:
                # Skip invalid time differences
                continue
                
            # Calculate displacement vector (just for position part - first 3 elements)
            dx = np.array(poses[i][:3]) - np.array(poses[i-1][:3])
            
            # Calculate velocity in m/s
            velocity = np.linalg.norm(dx) / dt
            
            velocities.append(velocity)
            time_differences.append(dt)
        
        # Initialize segment tracking
        active_segments = []
        current_segment = []
        current_segment_timestamps = []
        stationary_count = 0
        
        # Process each position with its velocity
        for i in range(len(poses)):
            # Handle first point
            if i == 0:
                current_segment.append(poses[i])
                current_segment_timestamps.append(tracked_timestamps[i])
                continue
                
            # Get velocity for this point (velocity at index i-1 corresponds to the motion from i-1 to i)
            if i-1 < len(velocities):
                velocity = velocities[i-1]
            else:
                # No velocity data available, add point and continue
                current_segment.append(poses[i])
                current_segment_timestamps.append(tracked_timestamps[i])
                continue
            
            # Check if this point is moving
            is_moving = velocity >= self.motion_velocity_threshold
            
            if is_moving:
                # Add point to current segment and reset stationary counter
                current_segment.append(poses[i])
                current_segment_timestamps.append(tracked_timestamps[i])
                stationary_count = 0
            else:
                # Point is stationary
                stationary_count += 1
                
                # Still include stationary points up to max_stationary_frames
                if stationary_count <= self.max_stationary_frames:
                    current_segment.append(poses[i])
                    current_segment_timestamps.append(tracked_timestamps[i])
                else:
                    # Too many stationary frames - end current segment if long enough
                    if len(current_segment) >= self.min_segment_frames:
                        # Store current segment
                        # 修复可能出现的IndexError，更安全地从bbox_corners_sequence获取数据
                        current_bbox_segment = []
                        for t in current_segment_timestamps:
                            if t in tracked_timestamps:
                                try:
                                    idx = tracked_timestamps.index(t)
                                    if 0 <= idx < len(bbox_corners_sequence):
                                        current_bbox_segment.append(bbox_corners_sequence[idx])
                                except (ValueError, IndexError) as e:
                                    print(f"Warning: Failed to get bbox corners for timestamp {t}: {e}")
                        
                        active_segments.append((current_segment.copy(), current_segment_timestamps.copy(), current_bbox_segment.copy()))
                    
                    # Start a new segment with this point
                    current_segment = [poses[i]]
                    current_segment_timestamps = [tracked_timestamps[i]]
                    stationary_count = 0
        
        # Add the final segment if it's long enough
        if len(current_segment) >= self.min_segment_frames:
            current_bbox_segment = []
            for t in current_segment_timestamps:
                if t in tracked_timestamps:
                    try:
                        idx = tracked_timestamps.index(t)
                        if 0 <= idx < len(bbox_corners_sequence):
                            current_bbox_segment.append(bbox_corners_sequence[idx])
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Failed to get bbox corners for timestamp {t}: {e}")
            
            active_segments.append((current_segment, current_segment_timestamps, current_bbox_segment))
        
        # If no segments found or all too short, return the original trajectory
        if not active_segments:
            return [(poses, tracked_timestamps, bbox_corners_sequence)]
        
        # Return all active segments in their original chronological order
        return active_segments
    
    def get_scene_pointcloud(self):
        """
        Return the scene pointcloud if available.
        
        Returns:
            Numpy array of point coordinates [N, 3] or None if not available
        """
        return self.pointcloud

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Get a trajectory sample.
        
        Args:
            idx: Index of the trajectory
        
        Returns:
            dict: Trajectory data including poses (positions+orientations), attention mask, and basic metadata
        """
        item = self.trajectories[idx]
        trajectory_data = item['trajectory_data']
        metadata = item['metadata']
        
        # Apply transforms if any
        if self.transform:
            trajectory_data = self.transform(trajectory_data)
        
        # Create the result with minimal metadata and split trajectories
        result = {
            'poses': trajectory_data['poses'],  # [N, 9] tensor with [x,y,z,r6d_1,r6d_2,r6d_3,r6d_4,r6d_5,r6d_6]
            'attention_mask': trajectory_data['attention_mask'],
            'object_type': metadata['prototype_name'],
            'object_name': metadata['name'],
            'object_id': metadata['id'],
            'object_category': metadata['category'],
            'object_subcategory': metadata['subcategory'],
            'sequence': metadata['sequence'],
            'is_displacement': metadata.get('is_displacement', False)
        }
        
        # Include trajectory-specific point cloud if available
        if 'trajectory_specific_pointcloud' in item:
            result['trajectory_specific_pointcloud'] = item['trajectory_specific_pointcloud']
        


        # Add segment index if available in metadata
        if 'segment_idx' in metadata:
            result['segment_idx'] = metadata['segment_idx']

        # Retrieve bbox_corners from trajectory_data and add to result
        if 'bbox_corners' in trajectory_data:
            result['bbox_corners'] = trajectory_data['bbox_corners']
        else:
            # Fallback if not present, though it should be due to _process_trajectory_segment
            result['bbox_corners'] = torch.zeros(self.trajectory_length, 8, 3, dtype=torch.float32)

        return result
    
    def get_trajectory_specific_pointcloud(self, idx):
        """
        Get the trajectory-specific pointcloud for a specific trajectory.
        
        Args:
            idx: Index of the trajectory
            
        Returns:
            Numpy array of point cloud coordinates or None if not available
        """
        if idx < 0 or idx >= len(self.trajectories):
            print(f"Error: Index {idx} out of range [0, {len(self.trajectories)-1}]")
            return None
            
        item = self.trajectories[idx]
        if 'trajectory_specific_pointcloud' in item:
            return item['trajectory_specific_pointcloud']
        else:
            print(f"Warning: No trajectory-specific pointcloud found for trajectory {idx}")
            return None
