#!/usr/bin/env python3
# HD-EPIC dataset loader for trajectory diffusion transformer

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import glob
import json
import pickle
import random
from pathlib import Path
from tqdm import tqdm
import re
import traceback
import pandas as pd
import zipfile
from PIL import Image
import tempfile
import sys

# Import Aria tools for pointcloud processing
try:
    import projectaria_tools.core.mps as aria_mps
    from projectaria_tools.core.mps.utils import filter_points_from_confidence
    ARIA_TOOLS_AVAILABLE = True
except ImportError:
    print("Warning: projectaria_tools not available. Pointcloud loading will be disabled.")
    ARIA_TOOLS_AVAILABLE = False

def load_participant_pointcloud(participant: str, base_path: str, video_id: str, subsample_ratio: float = 0.02):
    """
    Load pointcloud for a specific HD-EPIC participant and video using MPS with proper HD-EPIC structure.
    
    Args:
        participant: Participant ID (e.g., "P01")
        base_path: Base path to HD-EPIC dataset
        video_id: Video ID to load pointcloud for (e.g., "P01-20240202-110250"). Required.
        subsample_ratio: Ratio of points to keep after subsampling
    Returns:
        numpy array of shape [N, 3] with point coordinates, or None if loading fails
    """
    if not ARIA_TOOLS_AVAILABLE:
        print(f"Cannot load pointcloud: projectaria_tools not available")
        return None
        
    # Use the proper HD-EPIC SLAM directory structure
    slam_base = f"{base_path}/SLAM-and-Gaze/{participant}/SLAM/multi"
    
    if not slam_base or not os.path.exists(slam_base):
        print(f"HD-EPIC SLAM directory not found: {slam_base}")
        return None
    
    print(f"Loading pointcloud from HD-EPIC SLAM directory: {slam_base}")
    
    try:
        # Load the VRS to SLAM mapping
        mapping_file = os.path.join(slam_base, "vrs_to_multi_slam.json")
        if not os.path.exists(mapping_file):
            print(f"  VRS to SLAM mapping file not found: {mapping_file}")
            return None
            
        with open(mapping_file, 'r') as f:
            vrs_to_slam_mapping = json.load(f)
        
        # Find the correct SLAM zip file for the video
        selected_zip = None
        selected_slam_idx = None
        
        # Try to find the mapping for this specific video
        vrs_key = f"{participant}/{video_id}.vrs"
        if vrs_key in vrs_to_slam_mapping:
            slam_idx = vrs_to_slam_mapping[vrs_key]
            zip_path = os.path.join(slam_base, f"{slam_idx}.zip")
            if os.path.exists(zip_path):
                selected_zip = zip_path
                selected_slam_idx = slam_idx
                print(f"  Found SLAM mapping: {video_id} -> {slam_idx}.zip")
            else:
                print(f"  SLAM zip file not found: {zip_path}")
        else:
            print(f"  No SLAM mapping found for video {video_id}")
                
        if selected_zip is None:
            print(f"  ERROR: No valid SLAM zip file found for video {video_id}")
            return None
        
        zip_name = os.path.basename(selected_zip)
        zip_size_mb = os.path.getsize(selected_zip) / (1024*1024)
        
        print(f"  Selected zip file: {zip_name} ({zip_size_mb:.1f} MB)")
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"  Extracting to temporary directory: {temp_dir}")
            
            # Extract the selected zip file
            with zipfile.ZipFile(selected_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            print(f"  ✓ Extracted {zip_name}")
            
            # Find the extracted slam directory
            extracted_dirs = glob.glob(os.path.join(temp_dir, "*/slam"))
            if not extracted_dirs:
                print(f"  ERROR: No slam directory found in extracted zip")
                return None
                
            slam_dir = extracted_dirs[0]
            parent_dir = os.path.dirname(slam_dir)
            
            print(f"  Found extracted SLAM directory: {slam_dir}")
            
            # Now use the standard MPS loader on the extracted directory
            paths_provider = aria_mps.MpsDataPathsProvider(parent_dir)
            data_paths = paths_provider.get_data_paths()
            if not data_paths:
                print(f"  No MPS data paths found in {parent_dir}")
                return None
                
            # Create MPS data provider
            mps_provider = aria_mps.MpsDataProvider(data_paths)
            
            # Check if semidense point cloud is available
            if not mps_provider.has_semidense_point_cloud():
                print(f"  No semidense point cloud available in {parent_dir}")
                return None
                
            # Get semidense point cloud
            point_cloud = mps_provider.get_semidense_point_cloud()
            if not point_cloud:
                print(f"  Failed to get semidense point cloud from {parent_dir}")
                return None
                
            print(f"  Loaded {len(point_cloud)} semidense points from {zip_name}")

            # Filter the point cloud using confidence thresholds
            inverse_distance_std_threshold = 0.001
            distance_std_threshold = 0.15
            point_cloud = filter_points_from_confidence(
                point_cloud, 
                inverse_distance_std_threshold,
                distance_std_threshold
            )
            print(f"  Filtered to {len(point_cloud)} confident points")
            
            # Convert to numpy array
            points = np.array([point.position_world for point in point_cloud])
            
            # Filter out zero points
            zero_points = np.all(np.abs(points) < 1e-6, axis=1)
            num_zero_points = np.sum(zero_points)
            
            if num_zero_points > 0:
                valid_points = points[~zero_points]
                print(f"    Filtered out {num_zero_points} zero points, {len(valid_points)} remaining")
                if len(valid_points) > 0:
                    points = valid_points
                else:
                    print(f"    Warning: All points are zeros after filtering")
                    return None
            
            # Subsample points
            if subsample_ratio < 1.0:
                subsample_factor = int(1.0 / subsample_ratio)
                if len(points) > subsample_factor:
                    indices = np.random.choice(len(points), size=len(points)//subsample_factor, replace=False)
                    points = points[indices]
                    print(f"  Subsampled to {len(points)} points (1/{subsample_factor} of original)")
            
            print(f"  Final pointcloud: {len(points)} points")
            return points
                    
    except Exception as e:
        print(f"Error with HD-EPIC SLAM pointcloud loader: {e}")
        import traceback
        traceback.print_exc()
    
    return None

# Scene-level bounding box functions from test_mask.py
def extract_3d_bbox_from_mask_improved(mask_data, mask_array, object_name=None):
    """Extract 3D bounding box from mask data using object-specific priors"""
    if mask_array is None:
        return None
        
    # Get 3D center from mask_data
    center_3d = mask_data.get('3d_location')
    if not center_3d or len(center_3d) != 3:
        return None
        
    center_3d = np.array(center_3d, dtype=np.float32)
    
    # Get 2D bbox from mask_data
    bbox_2d = mask_data.get('bbox', [0, 0, 1, 1])
    if len(bbox_2d) != 4:
        bbox_2d = [0, 0, 1, 1]
    
    # Calculate 2D bbox dimensions
    width_2d = abs(bbox_2d[2] - bbox_2d[0])   # pixels
    height_2d = abs(bbox_2d[3] - bbox_2d[1])  # pixels
    aspect_ratio = width_2d / max(height_2d, 1e-6)  # w/h ratio
    
    # Object-specific size priors (typical real-world sizes in meters)
    size_priors = {
        'mug': [0.08, 0.10, 0.08],           # 8cm x 10cm x 8cm
        'cup': [0.08, 0.09, 0.08],           # 8cm x 9cm x 8cm  
        'glass': [0.07, 0.12, 0.07],         # 7cm x 12cm x 7cm
        'bottle': [0.07, 0.25, 0.07],        # 7cm x 25cm x 7cm
        'milk_bottle': [0.09, 0.28, 0.09],   # 9cm x 28cm x 9cm
        'milk bottle': [0.09, 0.28, 0.09],   # Alternative spelling
        'coffee_capsule': [0.04, 0.02, 0.04], # 4cm x 2cm x 4cm
        'coffee capsule': [0.04, 0.02, 0.04], # Alternative spelling
        'capsule': [0.04, 0.02, 0.04],       # Short form
        'scissors': [0.20, 0.08, 0.02],      # 20cm x 8cm x 2cm
        'knife': [0.25, 0.02, 0.02],         # 25cm x 2cm x 2cm
        'spoon': [0.15, 0.03, 0.01],         # 15cm x 3cm x 1cm
        'fork': [0.18, 0.02, 0.01],          # 18cm x 2cm x 1cm
        'plate': [0.25, 0.02, 0.25],         # 25cm x 2cm x 25cm
        'bowl': [0.15, 0.08, 0.15],          # 15cm x 8cm x 15cm
        'orange': [0.08, 0.08, 0.08],        # 8cm x 8cm x 8cm
        'apple': [0.07, 0.07, 0.07],         # 7cm x 7cm x 7cm
        'banana': [0.15, 0.03, 0.03],        # 15cm x 3cm x 3cm
        'chopping_board': [0.30, 0.02, 0.20], # 30cm x 2cm x 20cm
        'chopping board': [0.30, 0.02, 0.20], # Alternative spelling
        'food_processor': [0.25, 0.30, 0.25], # 25cm x 30cm x 25cm
        'food processor': [0.25, 0.30, 0.25], # Alternative spelling
        'unknown': [0.10, 0.10, 0.10],       # Default 10cm cube
    }
    
    # Use provided object name or try to detect from mask_data
    if object_name:
        obj_name_lower = object_name.lower()
    else:
        obj_name_lower = 'unknown'
    
    # Try to find a matching size prior
    prior_size = None
    matched_key = 'unknown'
    
    for key, size in size_priors.items():
        if key.lower() in obj_name_lower or obj_name_lower in key.lower():
            prior_size = size
            matched_key = key
            break
    
    if prior_size is None:
        prior_size = size_priors['unknown']
    
    # Estimate scale based on distance and 2D bbox size
    # Assume typical camera parameters for HD-EPIC (rough estimates)
    focal_length_pixels = 1000  # Rough estimate for HD cameras
    distance_to_object = np.linalg.norm(center_3d)  # Distance from camera origin
    
    # Calculate expected 2D size if object was at this distance
    expected_width_2d = (prior_size[0] * focal_length_pixels) / max(distance_to_object, 0.1)
    expected_height_2d = (prior_size[1] * focal_length_pixels) / max(distance_to_object, 0.1)
    
    # Scale the prior based on actual vs expected 2D size
    scale_x = width_2d / max(expected_width_2d, 1e-6)
    scale_y = height_2d / max(expected_height_2d, 1e-6)
    scale_avg = (scale_x + scale_y) / 2.0
    
    # Apply scaling to prior size (but constrain to reasonable bounds)
    scale_factor = np.clip(scale_avg, 0.3, 3.0)  # Don't scale more than 3x or less than 0.3x
    
    estimated_size = [
        prior_size[0] * scale_factor,  # width
        prior_size[1] * scale_factor,  # height  
        prior_size[2] * scale_factor   # depth
    ]
    
    # Ensure minimum size (2cm minimum)
    min_size = 0.02
    size_3d = [max(s, min_size) for s in estimated_size]
    
    # Count mask pixels
    mask_pixels = np.sum(mask_array > 0) if mask_array is not None else 0
    
    return {
        'center': center_3d,
        'size': size_3d,
        'mask_pixels': mask_pixels,
        'object_type': matched_key,
        'original_object_name': object_name or 'unknown',
        'scale_factor': scale_factor,
        'distance': distance_to_object,
        'bbox_2d_pixels': [width_2d, height_2d],
        'prior_size': prior_size
    }

# Keep the old function name for backward compatibility  
def extract_3d_bbox_from_mask_simple(mask_data, mask_array):
    """Backward compatibility wrapper"""
    return extract_3d_bbox_from_mask_improved(mask_data, mask_array)

def get_all_scene_objects_with_temporal_positioning(video_id, reference_frame, annotations_path, mask_base_path):
    """Get ALL objects with temporally accurate positioning relative to reference frame"""
    print(f"Getting scene objects with temporal positioning for {video_id}, ref frame: {reference_frame}")
    
    # Load annotations
    assoc_info_path = os.path.join(annotations_path, "scene-and-object-movements", "assoc_info.json")
    mask_info_path = os.path.join(annotations_path, "scene-and-object-movements", "mask_info.json")
    
    if not os.path.exists(assoc_info_path) or not os.path.exists(mask_info_path):
        print(f"Annotation files not found: {assoc_info_path}, {mask_info_path}")
        return []
    
    with open(assoc_info_path, 'r') as f:
        assoc_info = json.load(f)
    with open(mask_info_path, 'r') as f:
        mask_info = json.load(f)
    
    if video_id not in mask_info:
        print(f"No mask info found for video {video_id}")
        return []
    
    # First, collect ALL masks for each object
    object_masks = {}  # obj_name -> [list of mask data]
    mask_dir = os.path.join(mask_base_path, video_id)
    
    for mask_id, mask_data in mask_info[video_id].items():
        frame_num = mask_data['frame_number']
        
        # Find object name
        obj_name = "unknown"
        for assoc_id, assoc_data in assoc_info[video_id].items():
            for track in assoc_data['tracks']:
                if mask_id in track['masks']:
                    obj_name = assoc_data["name"]
                    break
            if obj_name != "unknown":
                break
        
        # Get 3D position
        pos_3d = mask_data.get('3d_location')
        if pos_3d and len(pos_3d) == 3:
            # Check if mask file exists
            mask_path = os.path.join(mask_dir, f"{mask_id}.png")
            if os.path.exists(mask_path):
                # Load mask for bbox calculation
                try:
                    mask = Image.open(mask_path)
                    mask_array = np.array(mask)
                    
                    # Add to object's mask list
                    if obj_name not in object_masks:
                        object_masks[obj_name] = []
                    
                    object_masks[obj_name].append({
                        'mask_id': mask_id,
                        'object_name': obj_name,
                        'frame_number': frame_num,
                        '3d_location': pos_3d,
                        'bbox_2d': mask_data.get('bbox', [0,0,0,0]),
                        'mask_array': mask_array,
                        'mask_data': mask_data
                    })
                    
                except Exception as e:
                    print(f"Error loading mask {mask_path}: {e}")
                    continue
    
    # Now apply temporal logic to select best frame for each object
    scene_objects = []
    
    for obj_name, masks in object_masks.items():
        # Sort masks by frame number
        masks.sort(key=lambda x: x['frame_number'])
        
        selected_mask = None
        reasoning = ""
        
        if len(masks) == 1:
            # Only one frame available
            selected_mask = masks[0]
            reasoning = "only frame"
        else:
            # Multiple frames available - apply temporal logic
            first_frame = masks[0]['frame_number']
            second_frame = masks[1]['frame_number'] if len(masks) > 1 else None
            
            if second_frame and second_frame < reference_frame:
                # Use second frame (more recent position before reference)
                selected_mask = masks[1]
                reasoning = f"2nd frame ({second_frame}) before ref ({reference_frame})"
            elif first_frame > reference_frame:
                # Use first frame (object already moved when we see it)
                selected_mask = masks[0]
                reasoning = f"1st frame ({first_frame}) after ref ({reference_frame})"
            else:
                # Use first frame (initial state)
                selected_mask = masks[0]
                reasoning = f"1st frame ({first_frame}) as initial state"
        
        if selected_mask:
            selected_mask['temporal_reasoning'] = reasoning
            selected_mask['available_frames'] = [m['frame_number'] for m in masks]
            scene_objects.append(selected_mask)
    
    # Sort by frame number of selected masks
    scene_objects.sort(key=lambda x: x['frame_number'])
    
    print(f"Found {len(scene_objects)} objects with temporal positioning")
    return scene_objects

def create_scene_bboxes_3d(scene_objects):
    """Create 3D bounding boxes for all scene objects using improved size estimation"""
    scene_bboxes = []
    
    for obj in scene_objects:
        # Extract 3D bbox using the improved method with object name
        bbox_3d = extract_3d_bbox_from_mask_improved(
            obj['mask_data'], 
            obj['mask_array'], 
            object_name=obj.get('object_name', 'unknown')
        )
        
        if bbox_3d:
            # Create bbox entry compatible with dataset format
            bbox_entry = {
                'center': bbox_3d['center'],  # [x, y, z]
                'size': bbox_3d['size'],      # [w, h, d]
                'rotation_6d': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Identity rotation
                'object_name': obj['object_name'],
                'frame_number': obj['frame_number'],
                'temporal_reasoning': obj.get('temporal_reasoning', 'unknown'),
                'mask_pixels': bbox_3d['mask_pixels'],
                # Additional debug info from improved extraction
                'object_type_detected': bbox_3d['object_type'],
                'scale_factor': bbox_3d['scale_factor'],
                'distance': bbox_3d['distance'],
                'bbox_2d_pixels': bbox_3d['bbox_2d_pixels'],
                'prior_size': bbox_3d['prior_size']
            }
            
            scene_bboxes.append(bbox_entry)
    
    print(f"Created {len(scene_bboxes)} scene-level 3D bounding boxes with improved size estimation")
    
    # Print some debug info for the first few objects
    for i, bbox in enumerate(scene_bboxes[:3]):
        print(f"  Object {i+1}: {bbox['object_name']} -> {bbox['object_type_detected']}")
        print(f"    Size: {[f'{s:.3f}' for s in bbox['size']]} (scale: {bbox['scale_factor']:.2f})")
        print(f"    Distance: {bbox['distance']:.2f}m, 2D: {bbox['bbox_2d_pixels']}")
    
    return scene_bboxes

class HDEpicTrajectoryDataset(Dataset):
    """Dataset for loading and processing HD-EPIC trajectory data for DiT models"""
    
    def __init__(
        self,
        base_path: str,
        annotations_path: str = "/home/wiss/saroha/github/project_aria/diffusion-trial/hd-epic-annotations-main",
        participants: Optional[List[str]] = None,
        trajectory_length: int = 104,
        skip_frames: int = 5,
        max_objects: Optional[int] = None,
        load_pointcloud: bool = False,
        pointcloud_subsample: int = 1000,
        min_motion_threshold: float = 0.0,
        min_motion_percentile: float = 0.0,
        use_displacements: bool = True,
        normalize_data: bool = True,
        use_cache: bool = True,
        cache_dir: str = "/storage/user/saroha/neurips25/project-aria/hdepic/hand-dense",
        trajectory_source: str = "SLAM",  # 'SLAM' or 'HAND' for trajectory source
        annotations_subdir: str = "scene-and-object-movements",  # Subdirectory containing annotations
        use_hand_interpolation: bool = True,  # Whether to use MPS hand data for dense trajectories
        load_scene_bboxes: bool = True,  # Whether to load scene-level bounding boxes (dynamic from masks)
        mask_base_path: str = "/storage/user/saroha/datasets/hd-epic-downloader/hd-epic/HD-EPIC/Digital-Twin/hd_epic_association_masks/masks",  # Path to mask files
        max_scene_objects: int = 100,  # Maximum number of scene objects to include
        max_sequences: Optional[int] = None,  # Maximum number of sequences/videos to load per participant
        bbox_csv_path: Optional[str] = None,  # Path to CSV file with static object bounding boxes
        load_static_bboxes: bool = False,  # Whether to load static bboxes from CSV
        max_static_objects: int = 100,  # Maximum number of static objects
        max_dynamic_objects: int = 100,  # Maximum number of dynamic objects
    ):
        """
        Initialize the HD-EPIC Dataset according to the HD-EPIC README specification.
        """
        self.base_path = base_path
        self.annotations_path = annotations_path
        self.annotations_subdir = annotations_subdir
        self.trajectory_length = trajectory_length
        self.skip_frames = skip_frames
        self.max_objects = max_objects
        self.load_pointcloud = load_pointcloud
        self.pointcloud_subsample = pointcloud_subsample
        self.min_motion_threshold = min_motion_threshold
        self.min_motion_percentile = min_motion_percentile
        self.use_displacements = use_displacements
        self.normalize_data = normalize_data
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.trajectory_source = trajectory_source
        self.use_hand_interpolation = use_hand_interpolation
        self.load_scene_bboxes = load_scene_bboxes  # For dynamic bboxes from masks
        self.mask_base_path = mask_base_path
        self.max_scene_objects = max_scene_objects
        self.max_sequences = max_sequences
        self.bbox_csv_path = bbox_csv_path
        self.load_static_bboxes = load_static_bboxes
        self.max_static_objects = max_static_objects
        self.max_dynamic_objects = max_dynamic_objects
        
        # Construct the full annotations path with subdirectory
        self.full_annotations_path = os.path.join(self.annotations_path, self.annotations_subdir) if self.annotations_subdir else self.annotations_path
        
        # Find all participants if not specified
        if participants is None:
            self.participants = self._find_all_participants()
        else:
            self.participants = participants
        
        # Load static bboxes AFTER participants are set
        self.static_bboxes_by_participant = {}
        if self.load_static_bboxes and self.bbox_csv_path:
            self.static_bboxes_by_participant = self._load_static_bboxes_from_csv()
            
        print(f"Using {len(self.participants)} participants: {self.participants}")
        
        # Create cache directory if needed
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Build dataset index
        self.dataset_index = self._build_dataset_index()
        
        # Process data and build trajectories
        self.trajectories = self._process_trajectories()
        
        print(f"Dataset ready with {len(self.trajectories)} trajectories")
    
    def _load_static_bboxes_from_csv(self) -> Dict[str, List[Dict]]:
        """Load static object bounding boxes from CSV file.
        
        Returns:
            Dictionary mapping participant ID to list of bbox dictionaries
        """
        if not os.path.exists(self.bbox_csv_path):
            print(f"Warning: Static bbox CSV file not found: {self.bbox_csv_path}")
            return {}
            
        try:
            # Read CSV file
            df = pd.read_csv(self.bbox_csv_path)
            print(f"Loading static bboxes from CSV: {self.bbox_csv_path}")
            print(f"  Found {len(df)} static objects in CSV")
            
            # Group by participant
            static_bboxes_by_participant = {}
            
            for participant in self.participants:
                # Filter bboxes for this participant
                participant_df = df[df['participant'] == participant]
                
                if len(participant_df) == 0:
                    print(f"  No static bboxes found for participant {participant}")
                    continue
                    
                # Convert to bbox format
                participant_bboxes = []
                for _, row in participant_df.iterrows():
                    bbox = {
                        'center': [row['center_x'], row['center_y'], row['center_z']],
                        'size': [row['size_w'], row['size_h'], row['size_d']],
                        'rotation_6d': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Identity rotation
                        'object_name': row['category'],
                        'object_index': row['index'],
                        'volume': row['volume'],
                        'obj_file': row.get('obj_file', ''),
                        'is_static': True,  # Mark as static object
                    }
                    participant_bboxes.append(bbox)
                
                static_bboxes_by_participant[participant] = participant_bboxes
                print(f"  Loaded {len(participant_bboxes)} static bboxes for {participant}")
                
            return static_bboxes_by_participant
            
        except Exception as e:
            print(f"Error loading static bboxes from CSV: {e}")
            traceback.print_exc()
            return {}
    
    def _find_all_participants(self) -> List[str]:
        """Find all participant directories according to HD-EPIC structure"""
        # First, try to extract participants from assoc_info.json
        assoc_info_path = os.path.join(self.full_annotations_path, "assoc_info.json")
        if os.path.exists(assoc_info_path):
            try:
                with open(assoc_info_path, 'r') as f:
                    assoc_info = json.load(f)
                
                # Extract participants from video IDs
                participants = set()
                for video_id in assoc_info.keys():
                    match = re.search(r'(P\d+)', video_id)
                    if match:
                        participants.add(match.group(1))
                
                if participants:
                    return sorted(list(participants))
            except Exception as e:
                print(f"Error reading assoc_info.json: {e}")
        
        # Fallback to directory structure
        if os.path.exists(self.full_annotations_path):
            participant_dirs = [d for d in os.listdir(self.full_annotations_path) 
                            if os.path.isdir(os.path.join(self.full_annotations_path, d)) and d.startswith('P')]
            if participant_dirs:
                return sorted(participant_dirs)
        
        # If nothing works, return a default list
        print("Using default participant list, no participants found in data.")
        return ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"]
    
    def _build_dataset_index(self) -> List[Dict]:
        """
        Build an index of all videos according to HD-EPIC README structure
        """
        dataset_index = []
        
        # Check for HD-EPIC structured files
        assoc_info_path = os.path.join(self.full_annotations_path, "assoc_info.json")
        mask_info_path = os.path.join(self.full_annotations_path, "mask_info.json")
        
        if os.path.exists(assoc_info_path) and os.path.exists(mask_info_path):
            print(f"Found HD-EPIC annotation files, using structured processing")
            
            try:
                # Load the association info file to get video IDs
                with open(assoc_info_path, 'r') as f:
                    assoc_info = json.load(f)
                    
                # Get all video IDs
                video_ids = list(assoc_info.keys())
                print(f"Found {len(video_ids)} videos in assoc_info.json")
                
                # Filter videos by participant if participants are specified
                if self.participants:
                    filtered_video_ids = []
                    participant_counts = {}  # Track how many sequences per participant
                    for video_id in video_ids:
                        # Extract participant ID from video ID (format is typically PXX-date-time)
                        match = re.search(r'(P\d+)', video_id)
                        participant = match.group(1) if match else "unknown"
                        if participant in self.participants:
                            # Check if we've reached the max sequences for this participant
                            if self.max_sequences is not None:
                                current_count = participant_counts.get(participant, 0)
                                if current_count >= self.max_sequences:
                                    continue  # Skip this video
                                participant_counts[participant] = current_count + 1
                            filtered_video_ids.append(video_id)
                    video_ids = filtered_video_ids
                    if self.max_sequences is not None:
                        print(f"Filtered to {len(video_ids)} videos (max {self.max_sequences} per participant) for participants: {self.participants}")
                    else:
                        print(f"Filtered to {len(video_ids)} videos for specified participants: {self.participants}")
                
                # Create one entry per video
                for video_id in video_ids:
                    # Extract participant ID from video ID
                    match = re.search(r'(P\d+)', video_id)
                    participant = match.group(1) if match else "unknown"
                    
                    # Add to index
                    dataset_index.append({
                        'participant': participant,
                        'video_id': video_id,
                        'base_name': video_id,
                        'annotation_file': assoc_info_path,
                        'hdepic_format': True
                    })
                    
                print(f"Created dataset index with {len(dataset_index)} video entries from HD-EPIC files")
                return dataset_index
                
            except Exception as e:
                print(f"Error loading HD-EPIC annotation files: {e}")
        
        print("WARNING: Could not find HD-EPIC annotation files.")
        print(f"Expected files: {assoc_info_path}, {mask_info_path}")
        return []
    
    def _load_annotations(self, entry: Dict) -> Optional[List[Dict]]:
        """
        Load trajectory annotations from HD-EPIC JSON files according to README structure
        """
        # Check for the HD-EPIC structured files
        assoc_info_path = os.path.join(self.full_annotations_path, "assoc_info.json")
        mask_info_path = os.path.join(self.full_annotations_path, "mask_info.json")
        
        if os.path.exists(assoc_info_path) and os.path.exists(mask_info_path):
            trajectories = self._parse_hdepic_annotations(assoc_info_path, mask_info_path, entry)
            return trajectories if trajectories else None
        else:
            print(f"HD-EPIC annotation files not found: {assoc_info_path}, {mask_info_path}")
            return None

    def _parse_hdepic_annotations(self, assoc_info_path: str, mask_info_path: str, entry: Dict) -> List[Dict]:
        """
        Parse HD-EPIC trajectory data according to README structure:
        
        assoc_info.json structure:
        {
          "video_id": {
            "association_id": {
              "name": "string",
              "tracks": [
                {
                  "track_id": "string", 
                  "time_segment": [start_time, end_time],
                  "masks": ["string", ...]
                }
              ]
            }
          }
        }
        
        mask_info.json structure:
        {
          "video_id": {
            "mask_id": {
              "frame_number": integer,
              "3d_location": [x, y, z],
              "bbox": [xmin, ymin, xmax, ymax],
              "fixture": "string"
            }
          }
        }
        """
        try:
            # Load both annotation files
            with open(assoc_info_path, 'r') as f:
                assoc_info = json.load(f)
                
            with open(mask_info_path, 'r') as f:
                mask_info = json.load(f)
                
            # Get the specific video ID for this entry
            video_id = entry.get('video_id') or entry.get('base_name', '')
            participant = entry.get('participant', 'unknown')
            
            if not video_id or video_id not in assoc_info or video_id not in mask_info:
                print(f"Video {video_id} not found in annotation files")
                return []
                
            video_assocs = assoc_info[video_id]
            video_masks = mask_info[video_id]
            
            print(f"Processing video {video_id}: {len(video_assocs)} associations, {len(video_masks)} masks")
            
            # Extract trajectories from associations
            all_trajectory_data = []
            
            # Process each association (object)
            for assoc_id, assoc_data in video_assocs.items():
                obj_name = assoc_data.get('name', 'unknown')
                tracks = assoc_data.get('tracks', [])
                
                if not tracks:
                    continue
                
                # Process each track separately (don't combine them)
                for track_idx, track in enumerate(tracks):
                    track_trajectory, track_frame_numbers = self._extract_single_track_trajectory(track, video_masks, obj_name)
                    if track_trajectory is not None and len(track_trajectory) >= 2:
                        
                        # Apply hand interpolation if enabled
                        if self.use_hand_interpolation:
                            track_trajectory = self.create_dense_trajectory_with_hand_interpolation(
                                track_trajectory, track_frame_numbers, obj_name, participant, video_id
                            )
                        
                        all_trajectory_data.append({
                            'trajectory': track_trajectory,
                            'object_name': obj_name,
                            'type': 'object',
                            'association_id': assoc_id,
                            'track_idx': track_idx,
                            'first_frame': track_frame_numbers[0] if track_frame_numbers else None
                        })
                        print(f"  Extracted track {track_idx} for {obj_name}: {len(track_trajectory)} points")
            
            # Filter trajectories by length
            if all_trajectory_data:
                min_length = 2
                valid_trajectory_data = [traj_data for traj_data in all_trajectory_data 
                                       if len(traj_data['trajectory']) >= min_length]
                
                if valid_trajectory_data:
                    # Sort by length (longest first)
                    valid_trajectory_data.sort(key=lambda x: len(x['trajectory']), reverse=True)
                    
                    print(f"Video {video_id}: Found {len(valid_trajectory_data)} valid trajectories")
                    lengths = [len(t['trajectory']) for t in valid_trajectory_data]
                    print(f"  Trajectory lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
                    
                    # Generate dynamic scene-level bounding boxes if enabled
                    if self.load_scene_bboxes:
                        print(f"Generating dynamic scene-level bounding boxes for video {video_id}")
                        
                        # Generate different bboxes for each trajectory based on its time frame
                        for traj_data in valid_trajectory_data:
                            # Use this trajectory's first frame as reference
                            reference_frame = traj_data.get('first_frame')
                            
                            if reference_frame is not None:
                                # Get scene objects with temporal positioning for this specific trajectory
                                scene_objects = get_all_scene_objects_with_temporal_positioning(
                                    video_id, reference_frame, self.annotations_path, self.mask_base_path
                                )
                                
                                if scene_objects:
                                    # Create 3D bounding boxes for this trajectory's timeframe
                                    dynamic_scene_bboxes = create_scene_bboxes_3d(scene_objects)
                                    
                                    # Limit number of dynamic scene objects
                                    if len(dynamic_scene_bboxes) > self.max_dynamic_objects:
                                        dynamic_scene_bboxes = dynamic_scene_bboxes[:self.max_dynamic_objects]
                                    
                                    # Add dynamic scene bboxes to this specific trajectory
                                    traj_data['dynamic_scene_bboxes'] = dynamic_scene_bboxes
                                    
                                    print(f"  Added {len(dynamic_scene_bboxes)} dynamic scene bboxes to trajectory {traj_data.get('object_name', 'unknown')} (frame {reference_frame})")
                                else:
                                    print(f"  No scene objects found for trajectory at frame {reference_frame}")
                                    traj_data['dynamic_scene_bboxes'] = []
                            else:
                                print(f"  Could not determine reference frame for trajectory {traj_data.get('object_name', 'unknown')}")
                                traj_data['dynamic_scene_bboxes'] = []
                    
                    return valid_trajectory_data
                else:
                    print(f"Video {video_id}: All trajectories too short")
                    return []
            else:
                print(f"Video {video_id}: No trajectories found")
                return []
                
        except Exception as e:
            print(f"Error parsing HD-EPIC annotations for video {video_id}: {e}")
            traceback.print_exc()
            return []
    
    def _extract_single_track_trajectory(self, track, video_masks, obj_name):
        """
        Extract trajectory for a single track (no combining)
        Returns (trajectory, frame_numbers) tuple
        """
        track_points = []  # [(position, frame_number)]
        
        track_id = track.get('track_id', 'unknown')
        time_segment = track.get('time_segment', [0, 0])
        mask_ids = track.get('masks', [])
        
        # Process each mask in this track
        for mask_id in mask_ids:
            if mask_id in video_masks:
                mask_data = video_masks[mask_id]
                
                # Extract 3D position according to README structure
                position_3d = mask_data.get('3d_location')
                if position_3d is None or not isinstance(position_3d, list) or len(position_3d) != 3:
                    continue
                    
                try:
                    position_3d = [float(p) for p in position_3d]
                    # Filter out clearly invalid positions
                    if all(abs(p) < 1e-8 for p in position_3d):  # All zeros
                        continue
                    if any(abs(p) > 100 for p in position_3d):  # Extremely large values
                        continue
                except (ValueError, TypeError):
                    continue
                    
                frame_number = mask_data.get('frame_number', 0)
                
                # Add identity rotation (6D representation) since HD-EPIC doesn't provide rotation
                rot_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                pos_rot = position_3d + rot_6d
                
                track_points.append((pos_rot, frame_number))
        
        if len(track_points) < 2:
            return None, None
            
        # Sort by frame number to create temporal trajectory
        track_points.sort(key=lambda x: x[1])
        
        # Remove duplicate frames (keep first occurrence)
        unique_points = []
        prev_frame = None
        for pos_rot, frame_num in track_points:
            if frame_num != prev_frame:
                unique_points.append((pos_rot, frame_num))
                prev_frame = frame_num
                
        if len(unique_points) < 2:
            return None, None
            
        # Return the single track trajectory and frame numbers
        trajectory = np.array([data[0] for data in unique_points])
        frame_numbers = [data[1] for data in unique_points]
        
        return trajectory, frame_numbers

    def load_mps_hand_tracking(self, participant, video_id):
        """Load MPS 3D hand tracking data for dense trajectory interpolation"""
        mps_zip_path = os.path.join(self.base_path, "SLAM-and-Gaze", participant, "GAZE_HAND", f"mps_{video_id}_vrs.zip")
        csv_path = os.path.join(self.base_path, "SLAM-and-Gaze", participant, "GAZE_HAND", "wrist_and_palm_poses.csv")
        
        # Extract CSV if needed
        if not os.path.exists(csv_path) and os.path.exists(mps_zip_path):
            with zipfile.ZipFile(mps_zip_path, 'r') as zip_ref:
                # Find the CSV in the zip
                for file_name in zip_ref.namelist():
                    if file_name.endswith('wrist_and_palm_poses.csv'):
                        zip_ref.extract(file_name, os.path.dirname(csv_path))
                        extracted_path = os.path.join(os.path.dirname(csv_path), file_name)
                        os.rename(extracted_path, csv_path)
                        break
        
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            return None

    def load_video_timing(self, participant, video_id):
        """Load video frame timing to sync with MPS timestamps"""
        timing_path = os.path.join(self.base_path, "Videos", participant, f"{video_id}_mp4_to_vrs_time_ns.csv")
        if os.path.exists(timing_path):
            return pd.read_csv(timing_path)
        else:
            return None

    def sync_frame_to_mps_timestamp(self, frame_idx, video_timing_df):
        """Convert frame index to MPS timestamp"""
        if video_timing_df is not None and frame_idx < len(video_timing_df):
            # Convert nanoseconds to microseconds (MPS uses microseconds)
            return video_timing_df.iloc[frame_idx]['vrs_device_time_ns'] / 1000
        return None

    def find_nearest_mps_detection(self, target_timestamp, mps_df, hand_type='left'):
        """Find nearest MPS detection to target timestamp"""
        if mps_df is None or len(mps_df) == 0 or target_timestamp is None:
            return None
            
        confidence_col = f'{hand_type}_tracking_confidence'
        valid_detections = mps_df[mps_df[confidence_col] > 0]
        
        if len(valid_detections) == 0:
            return None
            
        # Find nearest timestamp
        time_diffs = np.abs(valid_detections['tracking_timestamp_us'] - target_timestamp)
        nearest_idx = time_diffs.idxmin()
        
        return valid_detections.loc[nearest_idx]

    def get_hand_position_at_timestamp(self, timestamp, mps_df, video_timing_df, hand_type='left'):
        """Get hand position at specific timestamp"""
        if timestamp is None or mps_df is None:
            return None
            
        mps_detection = self.find_nearest_mps_detection(timestamp, mps_df, hand_type)
        if mps_detection is None:
            return None
            
        # Get hand position (using wrist as primary tracking point)
        hand_pos = np.array([
            mps_detection[f'tx_{hand_type}_wrist_device'],
            mps_detection[f'ty_{hand_type}_wrist_device'], 
            mps_detection[f'tz_{hand_type}_wrist_device']
        ])
        
        return hand_pos

    def create_dense_trajectory_with_hand_interpolation(self, track_trajectory, track_frame_numbers, obj_name, participant, video_id):
        """Create dense trajectory by interpolating between HD-EPIC coordinates using hand tracking"""
        if not self.use_hand_interpolation or len(track_trajectory) < 2 or len(track_frame_numbers) < 2:
            return track_trajectory
            
        try:
            # Load MPS hand tracking data
            mps_df = self.load_mps_hand_tracking(participant, video_id)
            video_timing_df = self.load_video_timing(participant, video_id)
            
            if mps_df is None or video_timing_df is None:
                return track_trajectory  # Return original if no hand data
            
            # Use the frame numbers we already have
            start_frame = track_frame_numbers[0]
            end_frame = track_frame_numbers[-1]
            
            if start_frame >= end_frame:
                return track_trajectory  # Can't interpolate
                
            # Create dense trajectory
            dense_trajectory = []
            
            # Interpolate frames between start and end
            total_frames = end_frame - start_frame + 1
            
            for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                # Get interpolation ratio
                ratio = i / (total_frames - 1) if total_frames > 1 else 0
                
                # Get timestamp for this frame
                frame_timestamp = self.sync_frame_to_mps_timestamp(frame_idx, video_timing_df)
                
                if frame_timestamp is not None:
                    # Try to get hand position at this frame
                    hand_pos_left = self.get_hand_position_at_timestamp(frame_timestamp, mps_df, video_timing_df, 'left')
                    hand_pos_right = self.get_hand_position_at_timestamp(frame_timestamp, mps_df, video_timing_df, 'right')
                    
                    # Use hand position (no fallbacks - hand data must be available)
                    if hand_pos_left is not None:
                        position_3d = hand_pos_left
                    elif hand_pos_right is not None:
                        position_3d = hand_pos_right
                    else:
                        # Skip this frame if no hand data available
                        continue
                else:
                    # Skip this frame if no timestamp available
                    continue
                
                # Add identity rotation (6D representation)
                rot_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                pos_rot = np.concatenate([position_3d, rot_6d])
                dense_trajectory.append(pos_rot)
            
            if len(dense_trajectory) >= 2:
                # print(f"  ✅ Hand interpolation created {len(dense_trajectory)} dense points for {obj_name}")
                return np.array(dense_trajectory)
            else:
                return track_trajectory
                
        except Exception as e:
            print(f"⚠️  Hand interpolation failed for {obj_name}: {e}")
            return track_trajectory  # Return original trajectory on error

    def _extract_object_trajectory(self, tracks, video_masks, obj_name):
        """
        Extract trajectory for an object by combining all its tracks according to HD-EPIC structure
        (Legacy method - keeping for compatibility)
        """
        all_track_points = []  # [(position, frame_number, track_id)]
        
        for track in tracks:
            track_id = track.get('track_id', 'unknown')
            time_segment = track.get('time_segment', [0, 0])
            mask_ids = track.get('masks', [])
            
            # Process each mask in this track
            for mask_id in mask_ids:
                if mask_id in video_masks:
                    mask_data = video_masks[mask_id]
                    
                    # Extract 3D position according to README structure
                    position_3d = mask_data.get('3d_location')
                    if position_3d is None or not isinstance(position_3d, list) or len(position_3d) != 3:
                        continue
                        
                    try:
                        position_3d = [float(p) for p in position_3d]
                        # Filter out clearly invalid positions
                        if all(abs(p) < 1e-8 for p in position_3d):  # All zeros
                            continue
                        if any(abs(p) > 100 for p in position_3d):  # Extremely large values
                            continue
                    except (ValueError, TypeError):
                        continue
                        
                    frame_number = mask_data.get('frame_number', 0)
                    
                    # Add identity rotation (6D representation) since HD-EPIC doesn't provide rotation
                    rot_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                    pos_rot = position_3d + rot_6d
                    
                    all_track_points.append((pos_rot, frame_number, track_id))
        
        if len(all_track_points) < 2:
            return None
            
        # Sort by frame number to create temporal trajectory
        all_track_points.sort(key=lambda x: x[1])
        
        # Remove duplicate frames (keep first occurrence)
        unique_points = []
        prev_frame = None
        for pos_rot, frame_num, track_id in all_track_points:
            if frame_num != prev_frame:
                unique_points.append((pos_rot, frame_num, track_id))
                prev_frame = frame_num
                
        if len(unique_points) < 2:
            return None
            
        # Return the combined trajectory
        return np.array([data[0] for data in unique_points]) 

    def _extract_trajectories(self, raw_trajectory: np.ndarray) -> List[np.ndarray]:
        """
        Extract fixed-length trajectories from raw trajectory data without artificial extension
        """
        if raw_trajectory is None or len(raw_trajectory) < 2:
            return []
            
        # Ensure we have at least position data
        if isinstance(raw_trajectory, list):
            raw_trajectory = np.array(raw_trajectory)
            
        if len(raw_trajectory.shape) == 1:
            if len(raw_trajectory) >= 3:
                raw_trajectory = raw_trajectory.reshape(1, -1)
            else:
                return []
                
        if raw_trajectory.shape[1] < 3:
            return []
            
        # Add identity rotation if missing
        if raw_trajectory.shape[1] == 3:
            identity_rotation = np.array([[1, 0, 0, 0, 1, 0]])
            rotations = np.tile(identity_rotation, (len(raw_trajectory), 1))
            raw_trajectory = np.concatenate([raw_trajectory, rotations], axis=1)
            
        trajectories = []
        
        # Use trajectory as-is if shorter than or equal to desired length
        if len(raw_trajectory) <= self.trajectory_length:
            trajectories.append(raw_trajectory)
            return trajectories
        
        # Create overlapping windows for longer trajectories
        actual_skip = max(1, self.skip_frames)
        window_step = self.trajectory_length // 2
        
        for start_idx in range(0, len(raw_trajectory) - self.trajectory_length + 1, window_step):
            end_idx = min(start_idx + self.trajectory_length * actual_skip, len(raw_trajectory))
            traj_indices = list(range(start_idx, end_idx, actual_skip))
            traj_indices = traj_indices[:self.trajectory_length]
            
            if len(traj_indices) < 2:
                continue
                
            trajectory = raw_trajectory[traj_indices]
            
            # Calculate motion for filtering
            total_motion = 0
            for i in range(1, len(trajectory)):
                total_motion += np.linalg.norm(trajectory[i, :3] - trajectory[i-1, :3])
                
            # Apply motion threshold if set
            if self.min_motion_threshold > 0 and total_motion < self.min_motion_threshold:
                continue
            
            trajectories.append(trajectory)
        
        # Fallback if no windows were created
        if not trajectories and len(raw_trajectory) > self.trajectory_length:
            indices = np.linspace(0, len(raw_trajectory)-1, self.trajectory_length, dtype=int)
            trajectory = raw_trajectory[indices]
            trajectories.append(trajectory)
        elif not trajectories:
            trajectories.append(raw_trajectory)
        
        return trajectories
    
    def _process_trajectories(self) -> List[Dict]:
        """Process all trajectories in the dataset"""
        # Check cache
        cache_suffix = f"_{self.trajectory_source}" if self.trajectory_source else ""
        hand_suffix = "_hand_interp" if self.use_hand_interpolation else ""
        static_suffix = "_static" if self.load_static_bboxes else ""
        cache_file = os.path.join(self.cache_dir, f"hdepic_trajectories{cache_suffix}{hand_suffix}{static_suffix}_L{self.trajectory_length}_S{self.skip_frames}_DISP{self.use_displacements}_NORM{self.normalize_data}.pkl")
        
        if self.use_cache and os.path.exists(cache_file):
            print(f"Loading cached trajectories from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        all_trajectories = []
        motion_values = []
        
        # Process each recording
        for entry in tqdm(self.dataset_index, desc="Processing trajectories"):
            annotation_trajectory_data = self._load_annotations(entry)
            
            if annotation_trajectory_data is None or len(annotation_trajectory_data) == 0:
                continue
                
            # Process each trajectory from the annotation
            for raw_trajectory_idx, trajectory_data in enumerate(annotation_trajectory_data):
                if trajectory_data is None:
                    continue
                
                # Extract trajectory array and metadata
                if isinstance(trajectory_data, dict):
                    raw_trajectory = trajectory_data.get('trajectory')
                    object_name = trajectory_data.get('object_name', 'unknown')
                    trajectory_type = trajectory_data.get('type', 'unknown')
                    association_id = trajectory_data.get('association_id', 'unknown')
                    track_idx = trajectory_data.get('track_idx', 0)
                    scene_bboxes = trajectory_data.get('scene_bboxes', [])
                else:
                    raw_trajectory = trajectory_data
                    object_name = 'unknown'
                    trajectory_type = 'legacy'
                    association_id = 'unknown'
                    track_idx = 0
                
                if raw_trajectory is None or len(raw_trajectory) == 0:
                    continue
                    
                # Extract fixed-length trajectories
                trajectories = self._extract_trajectories(raw_trajectory)
                
                if not trajectories:
                    continue
            
                # Add metadata for each extracted trajectory
                for idx, trajectory in enumerate(trajectories):
                    # Calculate motion
                    total_motion = 0
                    for i in range(1, len(trajectory)):
                        total_motion += np.linalg.norm(trajectory[i, :3] - trajectory[i-1, :3])
                    
                    motion_values.append(total_motion)
                    
                    # Store original positions
                    first_position = trajectory[0].copy()
                    end_position = trajectory[-1].copy() if len(trajectory) > 0 else first_position.copy()
                
                    # Add to list with metadata
                    all_trajectories.append({
                        'positions': trajectory,
                        'participant': entry['participant'],
                        'base_name': entry.get('base_name', 'unknown'),
                        'video_id': entry.get('video_id', 'unknown'),
                        'trajectory_idx': idx,
                        'raw_trajectory_idx': raw_trajectory_idx,
                        'total_motion': total_motion,
                        'first_position': first_position,
                        'end_position': end_position,
                        'source': 'hdepic',
                        'object_name': object_name,
                        'trajectory_type': trajectory_type,
                        'association_id': association_id,
                        'track_idx': track_idx,
                        'dynamic_scene_bboxes': trajectory_data.get('dynamic_scene_bboxes', []),
                    })
        
        print(f"Extracted {len(all_trajectories)} trajectories")
        
        if not all_trajectories:
            print("WARNING: No trajectories were successfully extracted!")
            print("Check that your annotations path contains the HD-EPIC scene-and-object-movements directory")
            print(f"Current annotation path: {self.full_annotations_path}")
            return []
        
        # Filter by motion percentile if needed
        if self.min_motion_percentile > 0 and all_trajectories:
            motion_threshold = np.percentile(motion_values, self.min_motion_percentile)
            filtered_trajectories = [t for t in all_trajectories if t['total_motion'] >= motion_threshold]
            print(f"Filtered trajectories by motion percentile {self.min_motion_percentile}%: {len(filtered_trajectories)} remaining")
            all_trajectories = filtered_trajectories
        
        # Apply absolute motion threshold if needed
        if self.min_motion_threshold > 0 and all_trajectories:
            filtered_trajectories = [t for t in all_trajectories if t['total_motion'] >= self.min_motion_threshold]
            print(f"Filtered trajectories by motion threshold {self.min_motion_threshold}: {len(filtered_trajectories)} remaining")
            
            if not filtered_trajectories and all_trajectories:
                top_n = max(1, int(len(all_trajectories) * 0.1))
                all_trajectories.sort(key=lambda x: x['total_motion'], reverse=True)
                filtered_trajectories = all_trajectories[:top_n]
                print(f"Motion threshold too strict, keeping top {top_n} trajectories")
                
            all_trajectories = filtered_trajectories
        
        # Normalize trajectories if requested
        if self.normalize_data and all_trajectories:
            all_trajectories = self._normalize_trajectories(all_trajectories)
        
        # Convert to displacements if requested
        if self.use_displacements and all_trajectories:
            for trajectory in all_trajectories:
                positions = trajectory['positions']
                first_position = positions[0].copy()
                
                # Convert to displacements
                displacements = np.zeros_like(positions)
                displacements[0] = positions[0]
                
                for i in range(1, len(positions)):
                    displacements[i, :3] = positions[i, :3] - positions[i-1, :3]
                    displacements[i, 3:] = positions[i, 3:]
                
                trajectory['positions'] = displacements
                trajectory['first_position'] = first_position
        
        # Cache processed trajectories
        if self.use_cache and all_trajectories:
            print(f"Caching {len(all_trajectories)} trajectories to {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(all_trajectories, f)
        
        return all_trajectories
    
    def _normalize_trajectories(self, trajectories: List[Dict]) -> List[Dict]:
        """Normalize trajectories based on overall scene bounds"""
        # Extract all positions for normalization
        all_positions = np.vstack([t['positions'][:, :3] for t in trajectories])
        
        # Calculate min and max values
        min_vals = np.min(all_positions, axis=0)
        max_vals = np.max(all_positions, axis=0)
        
        # Calculate range and center
        data_range = max_vals - min_vals
        data_center = (min_vals + max_vals) / 2
        
        print(f"Normalizing trajectories (min: {min_vals}, max: {max_vals})")
        
        # Normalize to [-1, 1] range
        for trajectory in trajectories:
            positions = trajectory['positions']
            positions[:, :3] = 2 * (positions[:, :3] - data_center) / data_range
            trajectory['positions'] = positions
            
            # Also normalize first_position and end_position if present
            if 'first_position' in trajectory:
                first_pos = trajectory['first_position']
                first_pos[:3] = 2 * (first_pos[:3] - data_center) / data_range
                trajectory['first_position'] = first_pos
                
            if 'end_position' in trajectory:
                end_pos = trajectory['end_position']
                end_pos[:3] = 2 * (end_pos[:3] - data_center) / data_range
                trajectory['end_position'] = end_pos
            
            # Store normalization parameters
            trajectory['norm_center'] = data_center
            trajectory['norm_range'] = data_range
        
        return trajectories
    

    def __len__(self) -> int:
        """Return the number of trajectories in the dataset"""
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a trajectory item"""
        trajectory = self.trajectories[idx]
        
        
        # Convert positions to tensor
        positions_tensor = torch.tensor(trajectory['positions'], dtype=torch.float32)
        
        # Create attention mask
        attention_mask = torch.ones(len(positions_tensor), dtype=torch.float32)
        
        # Pad if necessary
        if len(positions_tensor) < self.trajectory_length:
            padding_length = self.trajectory_length - len(positions_tensor)
            padding = torch.zeros(padding_length, positions_tensor.shape[1], dtype=torch.float32)
            positions_tensor = torch.cat([positions_tensor, padding], dim=0)
            
            # Update attention mask
            attention_mask = torch.cat([
                torch.ones(len(trajectory['positions']), dtype=torch.float32),
                torch.zeros(padding_length, dtype=torch.float32)
            ], dim=0)
        
        # Create return dictionary
        result = {
            'positions': positions_tensor,
            'attention_mask': attention_mask,
            'participant': trajectory['participant'],
            'base_name': trajectory['base_name'],
            'video_id': trajectory.get('video_id', 'unknown'),
            'object_category': trajectory.get('object_name', self.trajectory_source),
            'is_displacement': self.use_displacements,
            'association_id': trajectory.get('association_id', 'unknown'),
            'track_idx': trajectory.get('track_idx', 0),
        }
        
        # Add first position (always include for compatibility with AriaTrajectoryDataset)
        result['first_position'] = torch.tensor(trajectory['first_position'], dtype=torch.float32)
        
        # Add end position for compatibility
        if 'end_position' in trajectory:
            result['end_position'] = torch.tensor(trajectory['end_position'], dtype=torch.float32)
        elif positions_tensor.shape[0] > 0:
            result['end_position'] = positions_tensor[-1]
            
        # Add normalization info for compatibility
        result['normalization'] = {
            'is_normalized': self.normalize_data,
            'scene_min': torch.tensor([-10.0, -10.0, -10.0], dtype=torch.float32),
            'scene_max': torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32),
            'scene_scale': torch.tensor([20.0, 20.0, 20.0], dtype=torch.float32),
        }
        
        # Get participant ID for static bboxes
        participant = trajectory['participant']
        
        # Initialize static bbox data
        static_bbox_tensor = torch.zeros((self.max_static_objects, 12), dtype=torch.float32)
        static_mask_tensor = torch.zeros(self.max_static_objects, dtype=torch.float32)
        static_categories = ["unknown"] * self.max_static_objects
        
        # Load static bboxes if available
        if self.load_static_bboxes and participant in self.static_bboxes_by_participant:
            static_bboxes = self.static_bboxes_by_participant[participant]
            num_static = min(len(static_bboxes), self.max_static_objects)
            
            for i, bbox in enumerate(static_bboxes[:num_static]):
                # Pack bbox data: [center(3), size(3), rotation_6d(6)]
                bbox_data = np.concatenate([
                    bbox['center'],      # [x, y, z]
                    bbox['size'],        # [w, h, d]  
                    bbox['rotation_6d']  # [6D rotation]
                ])
                
                static_bbox_tensor[i] = torch.tensor(bbox_data, dtype=torch.float32)
                static_mask_tensor[i] = 1.0  # Mark as valid
                static_categories[i] = bbox.get('object_name', 'unknown')
        
        # Initialize dynamic bbox data
        dynamic_bbox_tensor = torch.zeros((self.max_dynamic_objects, 12), dtype=torch.float32)
        dynamic_mask_tensor = torch.zeros(self.max_dynamic_objects, dtype=torch.float32)
        dynamic_categories = ["unknown"] * self.max_dynamic_objects
        
        # Load dynamic bboxes if available
        if 'dynamic_scene_bboxes' in trajectory and self.load_scene_bboxes:
            dynamic_bboxes = trajectory['dynamic_scene_bboxes']
            num_dynamic = min(len(dynamic_bboxes), self.max_dynamic_objects)
            
            for i, bbox in enumerate(dynamic_bboxes[:num_dynamic]):
                # Pack bbox data: [center(3), size(3), rotation_6d(6)]
                bbox_data = np.concatenate([
                    bbox['center'],      # [x, y, z]
                    bbox['size'],        # [w, h, d]  
                    bbox['rotation_6d']  # [6D rotation]
                ])
                
                dynamic_bbox_tensor[i] = torch.tensor(bbox_data, dtype=torch.float32)
                dynamic_mask_tensor[i] = 1.0  # Mark as valid
                dynamic_categories[i] = bbox.get('object_name', 'unknown')
        
        # Create combined bbox data (for backward compatibility)
        max_combined = min(self.max_static_objects + self.max_dynamic_objects, 100)
        combined_bbox_tensor = torch.zeros((100, 12), dtype=torch.float32)
        combined_mask_tensor = torch.zeros(100, dtype=torch.float32)
        combined_categories = ["unknown"] * 100
        
        # First add static objects
        num_static_valid = int(static_mask_tensor.sum().item())
        if num_static_valid > 0:
            combined_bbox_tensor[:num_static_valid] = static_bbox_tensor[:num_static_valid]
            combined_mask_tensor[:num_static_valid] = static_mask_tensor[:num_static_valid]
            combined_categories[:num_static_valid] = static_categories[:num_static_valid]
        
        # Then add dynamic objects
        num_dynamic_valid = int(dynamic_mask_tensor.sum().item())
        if num_dynamic_valid > 0 and num_static_valid < max_combined:
            num_to_add = min(num_dynamic_valid, max_combined - num_static_valid)
            combined_bbox_tensor[num_static_valid:num_static_valid+num_to_add] = dynamic_bbox_tensor[:num_to_add]
            combined_mask_tensor[num_static_valid:num_static_valid+num_to_add] = dynamic_mask_tensor[:num_to_add]
            combined_categories[num_static_valid:num_static_valid+num_to_add] = dynamic_categories[:num_to_add]
        
        # Add bbox data to result
        result['static_scene_bbox'] = static_bbox_tensor
        result['static_bbox_mask'] = static_mask_tensor
        result['static_bbox_categories'] = static_categories
        
        result['dynamic_scene_bbox'] = dynamic_bbox_tensor
        result['dynamic_bbox_mask'] = dynamic_mask_tensor
        result['dynamic_bbox_categories'] = dynamic_categories
        
        # For backward compatibility
        result['bbox_info'] = combined_bbox_tensor
        result['bbox_mask'] = combined_mask_tensor
        result['bbox_categories'] = combined_categories
        
        # Set identity rotation in 6D representation for all bounding boxes
        result['static_scene_bbox'][:, 6] = 1.0  # rx1 = 1
        result['static_scene_bbox'][:, 10] = 1.0  # ry2 = 1
        result['dynamic_scene_bbox'][:, 6] = 1.0  # rx1 = 1
        result['dynamic_scene_bbox'][:, 10] = 1.0  # ry2 = 1
        result['bbox_info'][:, 6] = 1.0  # rx1 = 1
        result['bbox_info'][:, 10] = 1.0  # ry2 = 1
        
        return result
    

class MultiSequenceHDEpicDataset(Dataset):
    """Dataset for loading trajectories from multiple HD-EPIC sequences/participants."""
    
    def __init__(
        self,
        base_path: str,
        annotations_path: str = "/home/wiss/saroha/github/project_aria/diffusion-trial/hd-epic-annotations-main",
        participants: Optional[List[str]] = None,
        trajectory_length: int = 32,
        skip_frames: int = 5,
        max_objects: Optional[int] = None,
        load_pointcloud: bool = False,
        pointcloud_subsample: int = 100,
        scene_encoder_output_dim: Optional[int] = None,
        min_motion_threshold: float = 0.0,
        min_motion_percentile: float = 0.0,
        use_displacements: bool = True,
        normalize_data: bool = True,
        use_cache: bool = True,
        cache_dir: str = "/storage/user/saroha/neurips25/project-aria/hdepic/hand-dense-multi",
        trajectory_source: str = "SLAM",
        annotations_subdir: str = "scene-and-object-movements",
        use_hand_interpolation: bool = True,
        split_by_participants: bool = True,  # Whether to create separate datasets per participant
        load_scene_bboxes: bool = False,  # Whether to load scene-level bounding boxes
        mask_base_path: str = "/storage/user/saroha/datasets/hd-epic-downloader/hd-epic/HD-EPIC/Digital-Twin/hd_epic_association_masks/masks",  # Path to mask files
        max_scene_objects: int = 100,  # Maximum number of scene objects to include
        bbox_csv_path: Optional[str] = None,  # Path to CSV file with static object bounding boxes
        load_static_bboxes: bool = False,  # Whether to load static bboxes from CSV
        max_static_objects: int = 100,  # Maximum number of static objects
        max_dynamic_objects: int = 100,  # Maximum number of dynamic objects
    ):
        """
        Initialize the multi-sequence HD-EPIC dataset.
        
        Args:
            base_path: Path to HD-EPIC dataset root
            annotations_path: Path to HD-EPIC annotations
            participants: List of participants to include (e.g., ["P01", "P02"]), None for all
            trajectory_length: Length of trajectories to extract
            skip_frames: Number of frames to skip between samples
            max_objects: Maximum number of objects per sequence (None for all)
            load_pointcloud: Whether to load pointcloud data
            pointcloud_subsample: Subsample factor for pointcloud
            min_motion_threshold: Minimum total motion threshold (in meters)
            min_motion_percentile: Filter trajectories below this percentile of motion
            use_displacements: Whether to use displacements instead of absolute positions
            normalize_data: Whether to normalize data using scene bounds
            use_cache: Whether to use caching for datasets
            cache_dir: Directory to store cached data
            trajectory_source: Source of trajectory data ("SLAM" or "HAND")
            annotations_subdir: Subdirectory containing annotations
            use_hand_interpolation: Whether to use MPS hand data for dense trajectories
            split_by_participants: Whether to create separate datasets per participant
            load_scene_bboxes: Whether to load scene-level bounding boxes
            mask_base_path: Path to mask files
            max_scene_objects: Maximum number of scene objects to include
            bbox_csv_path: Path to CSV file with static object bounding boxes
            load_static_bboxes: Whether to load static bboxes from CSV
            max_static_objects: Maximum number of static objects
            max_dynamic_objects: Maximum number of dynamic objects
        """
        self.base_path = base_path
        self.annotations_path = annotations_path
        self.participants = participants
        self.trajectory_length = trajectory_length
        self.skip_frames = skip_frames
        self.max_objects = max_objects
        self.load_pointcloud = load_pointcloud
        self.pointcloud_subsample = pointcloud_subsample
        self.min_motion_threshold = min_motion_threshold
        self.min_motion_percentile = min_motion_percentile
        self.use_displacements = use_displacements
        self.normalize_data = normalize_data
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.trajectory_source = trajectory_source
        self.annotations_subdir = annotations_subdir
        self.use_hand_interpolation = use_hand_interpolation
        self.split_by_participants = split_by_participants
        self.load_scene_bboxes = load_scene_bboxes
        self.mask_base_path = mask_base_path
        self.max_scene_objects = max_scene_objects
        self.bbox_csv_path = bbox_csv_path
        self.load_static_bboxes = load_static_bboxes
        self.max_static_objects = max_static_objects
        self.max_dynamic_objects = max_dynamic_objects
        
        # This will store all individual datasets
        self.individual_datasets = []
        
        # Map from global index to (dataset_index, local_index)
        self.index_map = []
        
        # Track which sequences/participants were successfully loaded
        self.loaded_sequences = []
        
        # Load all sequences
        self._load_sequences()
        
        print(f"MultiSequenceHDEpicDataset initialized with {len(self.loaded_sequences)} sequences")
        print(f"Total trajectories: {len(self)}")
    
    def _load_sequences(self):
        """Load all specified sequences/participants."""
        
        if self.split_by_participants:
            # Create separate datasets for each participant
            self._load_by_participants()
        else:
            # Create one dataset with all participants
            self._load_single_dataset()
    
    def _load_by_participants(self):
        """Create separate datasets for each participant."""
        
        # First discover all participants if not specified
        if self.participants is None:
            temp_dataset = HDEpicTrajectoryDataset(
                base_path=self.base_path,
                annotations_path=self.annotations_path,
                trajectory_length=2,  # Small size just to discover participants
                use_cache=False,
                annotations_subdir=self.annotations_subdir
            )
            all_participants = temp_dataset.participants
        else:
            all_participants = self.participants
        
        print(f"Loading HD-EPIC data for {len(all_participants)} participants...")
        
        for i, participant in enumerate(all_participants):
            try:
                print(f"Loading participant {i+1}/{len(all_participants)}: {participant}")
                
                # Create dataset for this participant with unique cache directory
                participant_cache_dir = os.path.join(self.cache_dir, participant)
                
                dataset = HDEpicTrajectoryDataset(
                    base_path=self.base_path,
                    annotations_path=self.annotations_path,
                    participants=[participant],  # Only this participant
                    trajectory_length=self.trajectory_length,
                    skip_frames=self.skip_frames,
                    max_objects=self.max_objects,
                    load_pointcloud=self.load_pointcloud,
                    pointcloud_subsample=self.pointcloud_subsample,
                    min_motion_threshold=self.min_motion_threshold,
                    min_motion_percentile=self.min_motion_percentile,
                    use_displacements=self.use_displacements,
                    normalize_data=self.normalize_data,
                    use_cache=self.use_cache,
                    cache_dir=participant_cache_dir,
                    trajectory_source=self.trajectory_source,
                    annotations_subdir=self.annotations_subdir,
                    use_hand_interpolation=self.use_hand_interpolation,
                    load_scene_bboxes=self.load_scene_bboxes,
                    mask_base_path=self.mask_base_path,
                    max_scene_objects=self.max_scene_objects,
                    bbox_csv_path=self.bbox_csv_path,
                    load_static_bboxes=self.load_static_bboxes,
                    max_static_objects=self.max_static_objects,
                    max_dynamic_objects=self.max_dynamic_objects,
                )
                
                # If the dataset has trajectories, add it to our collection
                if len(dataset) > 0:
                    self.individual_datasets.append(dataset)
                    self.loaded_sequences.append(participant)
                    
                    # Update the index map
                    dataset_idx = len(self.individual_datasets) - 1
                    for local_idx in range(len(dataset)):
                        self.index_map.append((dataset_idx, local_idx))
                    
                    print(f"  Added {len(dataset)} trajectories from {participant}")
                else:
                    print(f"  Skipping {participant} - no trajectories found")
                    
            except Exception as e:
                print(f"Error loading participant {participant}: {e}")
                traceback.print_exc()
    
    def _load_single_dataset(self):
        """Create one dataset with all participants."""
        try:
            print(f"Loading HD-EPIC data for all participants as single dataset...")
            
            dataset = HDEpicTrajectoryDataset(
                base_path=self.base_path,
                annotations_path=self.annotations_path,
                participants=self.participants,
                trajectory_length=self.trajectory_length,
                skip_frames=self.skip_frames,
                max_objects=self.max_objects,
                load_pointcloud=self.load_pointcloud,
                pointcloud_subsample=self.pointcloud_subsample,
                min_motion_threshold=self.min_motion_threshold,
                min_motion_percentile=self.min_motion_percentile,
                use_displacements=self.use_displacements,
                normalize_data=self.normalize_data,
                use_cache=self.use_cache,
                cache_dir=self.cache_dir,
                trajectory_source=self.trajectory_source,
                annotations_subdir=self.annotations_subdir,
                use_hand_interpolation=self.use_hand_interpolation,
                load_scene_bboxes=self.load_scene_bboxes,
                mask_base_path=self.mask_base_path,
                max_scene_objects=self.max_scene_objects,
                bbox_csv_path=self.bbox_csv_path,
                load_static_bboxes=self.load_static_bboxes,
                max_static_objects=self.max_static_objects,
                max_dynamic_objects=self.max_dynamic_objects,
            )
            
            # If the dataset has trajectories, add it to our collection
            if len(dataset) > 0:
                self.individual_datasets.append(dataset)
                self.loaded_sequences.append("all_participants")
                
                # Update the index map
                for local_idx in range(len(dataset)):
                    self.index_map.append((0, local_idx))
                
                print(f"  Added {len(dataset)} trajectories from all participants")
            else:
                print(f"  No trajectories found")
                
        except Exception as e:
            print(f"Error loading HD-EPIC dataset: {e}")
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
        
        # Add the sequence info
        sequence_name = self.loaded_sequences[dataset_idx]
        sample['sequence_name'] = sequence_name
        sample['dataset_idx'] = dataset_idx
        
        return sample
    

    
    def get_sequence_count(self):
        """Get the number of loaded sequences/participants."""
        return len(self.loaded_sequences)
    
    def get_sequences_info(self):
        """Get information about loaded sequences."""
        info = {}
        for i, (dataset, sequence_name) in enumerate(zip(self.individual_datasets, self.loaded_sequences)):
            info[sequence_name] = {
                'dataset_idx': i,
                'num_trajectories': len(dataset),
                'participants': getattr(dataset, 'participants', [sequence_name])
            }
        return info


# Example usage
if __name__ == "__main__":
    # Example 1: Single dataset with hand interpolation
    print("=" * 60)
    print("EXAMPLE 1: Single HD-EPIC Dataset with Hand Interpolation")
    print("=" * 60)
    
    dataset = HDEpicTrajectoryDataset(
        base_path="/storage/user/saroha/datasets/hd-epic-downloader/hd-epic/HD-EPIC/",
        annotations_path="/storage/user/saroha/datasets/hd-epic-annotations-main",
        trajectory_length=32,
        skip_frames=5,
        use_displacements=True,
        normalize_data=True,
        use_cache=False,  # 🔧 Disable cache to regenerate data with scene_bboxes
        trajectory_source='SLAM',
        annotations_subdir='scene-and-object-movements',
        use_hand_interpolation=True,  # Enable dense trajectories with hand data
        participants=["P01"],  # Test with one participant first
        max_sequences=1,  # Only load the first sequence
        load_scene_bboxes=True,  # 🔧 Explicitly enable dynamic scene bboxes
        bbox_csv_path="/usr/stud/zehu/project/GIMO_ADT/dataset/all_objects_bbox.csv",  # 🆕 Static bbox CSV
        load_static_bboxes=True,  # 🆕 Enable static bbox loading
        max_static_objects=50,  # 🆕 Limit static objects
        max_dynamic_objects=50,  # 🆕 Limit dynamic objects
        cache_dir="./hdepic"
    )
    
    
    print(f"Single dataset contains {len(dataset)} trajectories")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample positions shape: {sample['positions'].shape}")
        print(f"Sample first_position shape: {sample['first_position'].shape}")
        print(f"Object name: {sample['object_category']}")
        print(f"Video ID: {sample['video_id']}")
        print(f"Track ID: {sample['track_idx']}")
        
        # 🆕 Show bbox information
        print(f"\n🆕 Bounding Box Information:")
        print(f"Static bboxes: {int(sample['static_bbox_mask'].sum())} objects")
        print(f"Dynamic bboxes: {int(sample['dynamic_bbox_mask'].sum())} objects")
        print(f"Combined bboxes: {int(sample['bbox_mask'].sum())} objects (backward compatible)")
        
        # Show first few static objects
        num_static = int(sample['static_bbox_mask'].sum())
        if num_static > 0:
            print(f"\nFirst 3 static objects:")
            for i in range(min(3, num_static)):
                print(f"  - {sample['static_bbox_categories'][i]}")
        
        # Show first few dynamic objects
        num_dynamic = int(sample['dynamic_bbox_mask'].sum())
        if num_dynamic > 0:
            print(f"\nFirst 3 dynamic objects:")
            for i in range(min(3, num_dynamic)):
                print(f"  - {sample['dynamic_bbox_categories'][i]}")
    
    # Example 2: Multi-sequence dataset 
    # print("\n" + "=" * 60)
    # print("EXAMPLE 2: Multi-Sequence HD-EPIC Dataset")
    # print("=" * 60)
    
    # multi_dataset = MultiSequenceHDEpicDataset(
    #     base_path="/storage/user/saroha/datasets/hd-epic-downloader/hd-epic/HD-EPIC/",
    #     annotations_path="/home/wiss/saroha/github/project_aria/diffusion-trial/hd-epic-annotations-main",
    #     trajectory_length=32,
    #     skip_frames=5,
    #     use_displacements=True,
    #     normalize_data=True,
    #     use_cache=True,
    #     trajectory_source='SLAM',
    #     annotations_subdir='scene-and-object-movements',
    #     use_hand_interpolation=True,
    #     participants=["P01", "P02"],  # Multiple participants
    #     split_by_participants=True,   # Create separate datasets per participant
    #     load_scene_bboxes=True,
    #     mask_base_path="/storage/user/saroha/datasets/hd-epic-downloader/hd-epic/HD-EPIC/Digital-Twin/hd_epic_association_masks/masks",
    #     max_scene_objects=100,
    # )
    
    # print(f"Multi-sequence dataset contains {len(multi_dataset)} trajectories")
    # print(f"Number of sequences: {multi_dataset.get_sequence_count()}")
    # print(f"Sequences info: {multi_dataset.get_sequences_info()}")
    
    # if len(multi_dataset) > 0:
    #     sample = multi_dataset[0]
    #     print(f"Sample positions shape: {sample['positions'].shape}")
    #     print(f"Sample first_position shape: {sample['first_position'].shape}")
    #     print(f"Object name: {sample['object_category']}")
    #     print(f"Sequence name: {sample['sequence_name']}")
    #     print(f"Dataset index: {sample['dataset_idx']}")
        
    #     # Test DataLoader compatibility
    #     from torch.utils.data import DataLoader
    #     dataloader = DataLoader(multi_dataset, batch_size=4, shuffle=True, num_workers=2)
        
    #     # Test batch loading
    #     for batch in dataloader:
    #         print(f"Batch positions shape: {batch['positions'].shape}")
    #         print(f"Batch keys: {list(batch.keys())}")
    #         print(f"Batch participants: {batch['participant']}")
    #         break
    # else:
    #     print("Dataset is empty! Check the annotations path and subdirectory.")
    #     print("Make sure assoc_info.json and mask_info.json exist in:")
    #     print("  annotations_path/scene-and-object-movements/") 