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
        load_pointcloud: bool = False,
        pointcloud_subsample: int = 100,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        min_motion_threshold: float = 0.0,  # Minimum motion threshold (in meters)
        min_motion_percentile: float = 0.0,  # Filter trajectories below this percentile of motion
        use_displacements: bool = False,  # Whether to use displacements instead of absolute positions
        detect_motion_segments: bool = True,  # Whether to detect and extract active motion segments
        motion_velocity_threshold: float = 0.05,  # Threshold in m/s for detecting active motion
        min_segment_frames: int = 5,  # Minimum number of frames for a valid motion segment
        max_stationary_frames: int = 3,   # Maximum consecutive stationary frames allowed in a motion segment
        normalize_data: bool = True       # Whether to normalize data using scene bounds
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
            normalize_data: Whether to normalize data using scene bounds (if config not provided)
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
            self.normalize_data = getattr(config, 'normalize_data', normalize_data) # Get normalize_data from config
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
            self.normalize_data = normalize_data # Use passed parameter if no config
        
        # Force load_pointcloud to True when normalize_data is True for maximum precision
        self.load_pointcloud = self.load_pointcloud or self.normalize_data

        # These parameters don't typically come from config
        self.max_objects = max_objects
        self.transform = transform
        self.pointcloud = None
        self.trajectories = []
        
        # Use provided cache_dir if not None, otherwise use config.save_path if available
        if cache_dir is not None:
            self.cache_dir = cache_dir
        elif config is not None and hasattr(config, 'save_path'):
            self.cache_dir = os.path.join(config.save_path, 'trajectory_cache')
        else:
            self.cache_dir = './trajectory_cache'  # Default fallback
        
        # Create cache directory if it doesn't exist
        if self.use_cache and not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir)
                print(f"Created cache directory: {self.cache_dir}")
            except Exception as e:
                print(f"Warning: Could not create cache directory: {e}")
                self.use_cache = False
        
        print(f"Loading ADT sequence from: {sequence_path}")
        print(f"Data Normalization: {'Enabled' if self.normalize_data else 'Disabled'}")
        
        # Initialize scene bounds - these will be calculated if normalize_data=True and pointcloud is loaded
        self.scene_min = np.zeros(3, dtype=np.float32)
        self.scene_max = np.ones(3, dtype=np.float32)
        self.scene_scale = np.ones(3, dtype=np.float32)
        if not self.normalize_data:
            print("Data normalization disabled, using identity transformation")
        
        # Check for cached data first
        if self.use_cache and self._load_from_cache():
            print(f"Loaded cached trajectories for {sequence_path}")
            # Bounds should be loaded from cache, print them
            print(f"  Using cached Scene bounds: min={self.scene_min}, max={self.scene_max}, scale={self.scene_scale}")
            return
            
        # Create data provider
        try:
            paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
            
            # Use the correct method name: get_datapaths instead of get_data_paths
            data_paths = paths_provider.get_datapaths(False)
            if data_paths is None:
                raise ValueError(f"Failed to get data paths from {sequence_path}")
                
            self.adt_provider = AriaDigitalTwinDataProvider(data_paths)
            
            # Load pointcloud if requested (or if needed for normalization)
            if self.load_pointcloud:
                self._load_pointcloud() # This will calculate bounds if normalize_data is True
            elif self.normalize_data:
                 print("Warning: Normalization enabled, but pointcloud loading is disabled. Attempting to load pointcloud anyway for bounds calculation.")
                 self._load_pointcloud()
                 if self.pointcloud is None:
                     print("Warning: Failed to load pointcloud for normalization. Using default bounds.")
                     self._set_default_scene_bounds()
            
            # Extract trajectories
            start_time = time.time()
            self._extract_trajectories()
            print(f"Trajectory extraction took {time.time() - start_time:.2f} seconds")
            
            # Bbox logic removed for GIMO version
            
            # Save to cache if caching is enabled
            if self.use_cache:
                self._save_to_cache()
            
        except Exception as e:
            print(f"Error loading ADT sequence: {e}")
            traceback.print_exc()
            raise
    
    def _get_cache_filename(self):
        """Generate a unique filename for the cache based on the dataset parameters."""
        # Create a unique identifier based on relevant parameters
        params = {
            'sequence_path': self.sequence_path,
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
            'normalize_data': self.normalize_data # Add normalize_data to cache key
        }
        
        # Generate hash from parameters
        params_str = str(params)
        hash_obj = hashlib.md5(params_str.encode())
        hash_str = hash_obj.hexdigest()
        
        # Extract sequence name from the path for more readable filenames
        sequence_name = os.path.basename(self.sequence_path)
        
        # Create filename
        return os.path.join(self.cache_dir, f"{sequence_name}_{hash_str}.pkl")
    
    def _save_to_cache(self):
        """Save trajectories and scene features to cache."""
        if not self.use_cache:
            return False
            
        cache_file = self._get_cache_filename()
        
        try:
            # Prepare data to save
            cache_data = {
                'trajectories': self.trajectories,
                'pointcloud': self.pointcloud,
                # Add normalization parameters to cache
                'scene_min': self.scene_min,
                'scene_max': self.scene_max,
                'scene_scale': self.scene_scale,
                'params': {
                    'sequence_path': self.sequence_path,
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
                    'normalize_data': self.normalize_data # Save normalize_data state
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
            
        cache_file = self._get_cache_filename()
        
        if not os.path.exists(cache_file):
            print(f"No cache file found at {cache_file}")
            return False
            
        try:
            # Load data from file
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Verify parameters match
            params = cache_data['params']
            if (params['sequence_path'] != self.sequence_path or
                params['trajectory_length'] != self.trajectory_length or
                params['skip_frames'] != self.skip_frames or
                params['max_objects'] != self.max_objects or
                params.get('use_displacements', False) != self.use_displacements or
                params.get('detect_motion_segments', True) != self.detect_motion_segments or
                params.get('motion_velocity_threshold', 0.05) != self.motion_velocity_threshold or
                params.get('min_segment_frames', 5) != self.min_segment_frames or
                params.get('max_stationary_frames', 3) != self.max_stationary_frames or
                params.get('min_motion_threshold', 0.0) != self.min_motion_threshold or
                params.get('min_motion_percentile', 0.0) != self.min_motion_percentile or
                params.get('pointcloud_subsample', 100) != self.pointcloud_subsample or
                params.get('normalize_data', True) != self.normalize_data): # Verify normalize_data
                print("Cache parameters don't match, regenerating trajectories")
                return False
                
            # Load data
            self.trajectories = cache_data['trajectories']
            self.pointcloud = cache_data['pointcloud']
            # Load normalization parameters from cache, with fallback for older caches
            self.scene_min = cache_data.get('scene_min', np.zeros(3, dtype=np.float32))
            self.scene_max = cache_data.get('scene_max', np.ones(3, dtype=np.float32))
            self.scene_scale = cache_data.get('scene_scale', np.ones(3, dtype=np.float32))
            # Ensure normalize_data flag is set correctly based on cache
            self.normalize_data = params.get('normalize_data', True)
            
            print(f"Loaded {len(self.trajectories)} trajectories from cache")
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

            # Calculate scene bounds for normalization only if normalize_data is True
            if self.normalize_data:
                if len(self.pointcloud) > 0:
                    # Calculate bounds from point cloud
                    self.scene_min = np.min(self.pointcloud, axis=0) 
                    self.scene_max = np.max(self.pointcloud, axis=0)

                    # Add padding to avoid points exactly at boundaries
                    padding = (self.scene_max - self.scene_min) * 0.05  # 5% padding
                    self.scene_min -= padding
                    self.scene_max += padding

                    # Calculate scale for normalization to [-1, 1] range
                    self.scene_scale = self.scene_max - self.scene_min
                    # Prevent division by zero for flat dimensions
                    self.scene_scale[self.scene_scale < 1e-6] = 1.0

                    print(f"Scene bounds calculated: min={self.scene_min}, max={self.scene_max}")
                    print(f"Scene scale: {self.scene_scale}")
                else:
                    print("Warning: Point cloud is empty, using default scene bounds")
                    self._set_default_scene_bounds()
            # If not normalizing, bounds remain identity matrix initialized in __init__
            
        except Exception as e:
            print(f"Error loading MPS point cloud: {e}")
            traceback.print_exc()
            self._set_default_scene_bounds() # Set default bounds on error
    
    def _set_default_scene_bounds(self):
        """Set default scene bounds if point cloud loading fails or normalization is enabled without a point cloud."""
        if self.normalize_data:
             print("Warning: Using default scene bounds [-10, 10] for normalization.")
             self.scene_min = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
             self.scene_max = np.array([10.0, 10.0, 10.0], dtype=np.float32)
             self.scene_scale = self.scene_max - self.scene_min
        else:
             # Keep identity bounds if not normalizing
             self.scene_min = np.zeros(3, dtype=np.float32)
             self.scene_max = np.ones(3, dtype=np.float32)
             self.scene_scale = np.ones(3, dtype=np.float32)

    def normalize_position(self, position):
        """Normalize position from world space to [-1, 1] range using scene bounds."""
        if not self.normalize_data:
            return np.asarray(position, dtype=np.float32)  # Return original if not normalizing
        
        position_array = np.asarray(position, dtype=np.float32)
        # Convert from world space to [0, 1] range
        normalized = (position_array - self.scene_min) / self.scene_scale
        # Convert from [0, 1] to [-1, 1] range
        normalized = normalized * 2.0 - 1.0
        return normalized

    def denormalize_position(self, normalized_position):
        """Denormalize position from [-1, 1] range back to world space using scene bounds."""
        if not self.normalize_data:
            return np.asarray(normalized_position, dtype=np.float32)  # Return as is if not normalized
        
        normalized_array = np.asarray(normalized_position, dtype=np.float32)
        # Convert from [-1, 1] to [0, 1] range
        denormalized = (normalized_array + 1.0) / 2.0
        # Convert from [0, 1] range to world space
        denormalized = denormalized * self.scene_scale + self.scene_min
        return denormalized

    # Included for consistency, though not used directly in GIMO dataset currently
    def normalize_size(self, size):
        """Normalize size/dimension using scene scale."""
        if not self.normalize_data:
            return np.asarray(size, dtype=np.float32)  # Return original if not normalizing
        
        size_array = np.asarray(size, dtype=np.float32)
        # Normalize size based on scene scale - result will be in [0, 1] range approximately
        normalized = size_array / self.scene_scale
        # Scale to [-1, 1] range for consistency with positions
        normalized = normalized * 2.0
        return normalized

    # Included for consistency, though not used directly in GIMO dataset currently
    def denormalize_size(self, normalized_size):
        """Denormalize size from normalized range back to world space."""
        if not self.normalize_data:
            return np.asarray(normalized_size, dtype=np.float32)  # Return as is if not normalized
        
        normalized_array = np.asarray(normalized_size, dtype=np.float32)
        # Scale back from [-1, 1] range-friendly format
        denormalized = normalized_array / 2.0
        # Convert back to world space scale
        denormalized = denormalized * self.scene_scale
        return denormalized
    
    def _calculate_trajectory_motion(self, positions, timestamps=None):
        """
        Calculate the total motion/displacement of a trajectory.
        
        Args:
            positions: List of position vectors
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
                for segment_idx, (positions, tracked_timestamps) in enumerate(segments):
                    if len(positions) < 2:
                        # Skip segments that are too short
                        continue
                        
                    # Process this segment
                    self._process_trajectory_segment(
                        positions, 
                        tracked_timestamps, 
                        instance_info, 
                        obj_id, 
                        trajectories_with_motion,
                        segment_idx=segment_idx
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
    
    def _process_trajectory_segment(self, positions, tracked_timestamps, instance_info, obj_id, trajectories_with_motion, segment_idx=None):
        """
        Process a single trajectory segment and add it to the trajectories list.
        
        Args:
            positions: List of position vectors
            tracked_timestamps: List of timestamps for each position
            instance_info: Object instance information
            obj_id: Object ID
            trajectories_with_motion: List to store processed trajectories
            segment_idx: Optional index of the segment (for multi-segment objects)
        """
        # Calculate trajectory motion metrics (using original positions before normalization)
        total_path_length, direct_displacement, max_displacement, avg_velocity = self._calculate_trajectory_motion(positions, tracked_timestamps)
        
        # Convert to numpy array and ensure consistent 2D shape [N, 3]
        positions_array = np.array(positions)
        if positions_array.ndim > 2:
            positions_array = positions_array.reshape(-1, 3)
        
        # Normalize positions if normalize_data is True
        if self.normalize_data:
            positions_array = self.normalize_position(positions_array)

        # Convert normalized (or original) positions to tensor
        positions_tensor = torch.tensor(positions_array, dtype=torch.float32)
        real_length = min(len(positions_tensor), self.trajectory_length)
        
        # Modified displacement calculation (applied on normalized positions if enabled)
        if self.use_displacements and real_length > 1:
            # Create a new tensor to store the first position and subsequent displacements
            modified_tensor = torch.zeros_like(positions_tensor)
            
            # Keep the first position as absolute coordinates (normalized or original)
            modified_tensor[0] = positions_tensor[0]
            
            # Calculate displacements for all positions after the first
            modified_tensor[1:real_length] = positions_tensor[1:real_length] - positions_tensor[:real_length-1]
            
            # Replace positions with this new representation (first position + displacements)
            positions_tensor = modified_tensor
            
            # No need to store first_position separately in metadata since it's part of the tensor now
            metadata_first_position = None
        else:
            metadata_first_position = None
        
        # Create attention mask (ones for real data, zeros for padding)
        attention_mask = torch.zeros(self.trajectory_length, dtype=torch.float32)
        attention_mask[:real_length] = 1.0
        
        # Pad the positions tensor if needed
        if len(positions_tensor) < self.trajectory_length:
            # Make sure padding has the same shape [M, 3]
            # Pad with zeros in the normalized space if normalizing, otherwise world space zeros
            padding = torch.zeros(self.trajectory_length - len(positions_tensor), 3, dtype=torch.float32)
            positions_tensor = torch.cat([positions_tensor, padding], dim=0)
        else:
            positions_tensor = positions_tensor[:self.trajectory_length]
        
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
            'original_length': len(positions),  # Store the length of the active segment
            'start_timestamp_ns': int(tracked_timestamps[0]) if tracked_timestamps else None,  # Store segment start timestamp
            # Store normalization parameters for denormalization later
            'normalization': {
                'is_normalized': self.normalize_data,
                'scene_min': self.scene_min.tolist(),
                'scene_max': self.scene_max.tolist(),
                'scene_scale': self.scene_scale.tolist()
            }
        }
        
        # Add segment index if provided
        if segment_idx is not None:
            metadata['segment_idx'] = segment_idx
        
        # Add first position if using displacements (not needed with our new approach)
        # if metadata_first_position is not None:
        #     metadata['first_position'] = metadata_first_position
        
        # Extract category and subcategory if available
        metadata['category'] = getattr(instance_info, 'category', 'unknown')
        metadata['subcategory'] = getattr(instance_info, 'subcategory', 'unknown')
        
        # Create trajectory data dict
        trajectory_data = {
            'positions': positions_tensor,
            'attention_mask': attention_mask,
        }
        
        # Create complete trajectory item
        trajectory_item = {
            'trajectory_data': trajectory_data,
            'metadata': metadata,
            'motion_value': total_path_length  # Store total path length for filtering (absolute distance moved)
        }
        
        trajectories_with_motion.append(trajectory_item)
    
    def _track_object_motion(self, object_id, timestamps):
        """
        Track an object's motion across a sequence.
        
        Args:
            object_id: ID of the object to track
            timestamps: List of timestamps to track at
            
        Returns:
            list: List of (positions, timestamps) tuples for all active motion segments,
                  or a list with a single tuple of the entire trajectory if motion detection is disabled
        """
        from collections import deque
        
        # Lists to store trajectory data
        positions = []
        tracked_timestamps = []
        
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
                    
                    # Store the position and timestamp
                    positions.append(pos)
                    tracked_timestamps.append(ts)
                except Exception as e:
                    if time_idx % 100 == 0 or time_idx == 1:
                        print(f"Warning: Could not get position for object {object_id} at time {ts}: {e}")
                    # Skip this timestamp
                    continue
        except Exception as e:
            print(f"Error tracking object {object_id}: {e}")
            traceback.print_exc()
        
        # If we didn't collect enough points or motion detection is disabled, return raw trajectory as a single segment
        if len(positions) < self.min_segment_frames or not self.detect_motion_segments:
            return [(positions, tracked_timestamps)]  # Return as a list with one segment
        
        # Calculate velocity between each pair of points
        velocities = []
        time_differences = []
        
        for i in range(1, len(positions)):
            # Calculate time difference in seconds
            dt = (tracked_timestamps[i] - tracked_timestamps[i-1]) / 1e9  # Convert ns to seconds
            
            if dt <= 0:
                # Skip invalid time differences
                continue
                
            # Calculate displacement vector
            dx = np.array(positions[i]) - np.array(positions[i-1])
            
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
        for i in range(len(positions)):
            # Handle first point
            if i == 0:
                current_segment.append(positions[i])
                current_segment_timestamps.append(tracked_timestamps[i])
                continue
                
            # Get velocity for this point (velocity at index i-1 corresponds to the motion from i-1 to i)
            if i-1 < len(velocities):
                velocity = velocities[i-1]
            else:
                # No velocity data available, add point and continue
                current_segment.append(positions[i])
                current_segment_timestamps.append(tracked_timestamps[i])
                continue
            
            # Check if this point is moving
            is_moving = velocity >= self.motion_velocity_threshold
            
            if is_moving:
                # Add point to current segment and reset stationary counter
                current_segment.append(positions[i])
                current_segment_timestamps.append(tracked_timestamps[i])
                stationary_count = 0
            else:
                # Point is stationary
                stationary_count += 1
                
                # Still include stationary points up to max_stationary_frames
                if stationary_count <= self.max_stationary_frames:
                    current_segment.append(positions[i])
                    current_segment_timestamps.append(tracked_timestamps[i])
                else:
                    # Too many stationary frames - end current segment if long enough
                    if len(current_segment) >= self.min_segment_frames:
                        # Store current segment
                        active_segments.append((current_segment.copy(), current_segment_timestamps.copy()))
                    
                    # Start a new segment with this point
                    current_segment = [positions[i]]
                    current_segment_timestamps = [tracked_timestamps[i]]
                    stationary_count = 0
        
        # Add the final segment if it's long enough
        if len(current_segment) >= self.min_segment_frames:
            active_segments.append((current_segment, current_segment_timestamps))
        
        # If no segments found or all too short, return the original trajectory
        if not active_segments:
            return [(positions, tracked_timestamps)]
        
        # # Print stats about all segments found
        # segment_lengths = [len(segment[0]) for segment in active_segments]
        # segment_durations = [(segment[1][-1] - segment[1][0]) / 1e9 for segment in active_segments]
        
        # # Get object name from instance_info
        # object_name = self.adt_provider.get_instance_info_by_id(object_id).name
        
        # print(f"Object {object_name}: Found {len(active_segments)} motion segments")
        # print(f"  Longest segment: {max(segment_lengths) if segment_lengths else 0} frames, {max(segment_durations) if segment_durations else 0:.2f}s")
        # print(f"  Total trajectory: {len(positions)} frames")
        
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
            dict: Trajectory data including positions, attention mask, and basic metadata
        """
        item = self.trajectories[idx]
        trajectory_data = item['trajectory_data']
        metadata = item['metadata']
        
        # Apply transforms if any
        if self.transform:
            trajectory_data = self.transform(trajectory_data)
        
        # Create the result with minimal metadata and split trajectories
        result = {
            'positions': trajectory_data['positions'],
            'attention_mask': trajectory_data['attention_mask'],
            'object_type': metadata['prototype_name'],
            'object_name': metadata['name'],
            'object_id': metadata['id'],
            'object_category': metadata['category'],
            'object_subcategory': metadata['subcategory'],
            'sequence': metadata['sequence'],
            'is_displacement': metadata.get('is_displacement', False)
        }
        
        # Add first position if using displacements (and not using the newer displacement representation)
        # if 'first_position' in metadata:
        #     result['first_position'] = torch.tensor(metadata['first_position'], dtype=torch.float32)
            
        # Add normalization parameters to the sample
        if 'normalization' in metadata:
             # Convert normalization numpy arrays to tensors for collation
             norm_info = metadata['normalization']
             result['normalization'] = {
                 'is_normalized': norm_info['is_normalized'],
                 'scene_min': torch.tensor(norm_info.get('scene_min', [0.0, 0.0, 0.0]), dtype=torch.float32),
                 'scene_max': torch.tensor(norm_info.get('scene_max', [1.0, 1.0, 1.0]), dtype=torch.float32),
                 'scene_scale': torch.tensor(norm_info.get('scene_scale', [1.0, 1.0, 1.0]), dtype=torch.float32)
             }
        else:
             # Include default normalization parameters if not found in metadata
             # (should only happen if loading old cache without normalization info)
             result['normalization'] = {
                 'is_normalized': self.normalize_data, # Use the dataset's current setting
                 'scene_min': torch.tensor(self.scene_min, dtype=torch.float32),
                 'scene_max': torch.tensor(self.scene_max, dtype=torch.float32),
                 'scene_scale': torch.tensor(self.scene_scale, dtype=torch.float32)
             }

        # Bounding box logic removed here

        return result
