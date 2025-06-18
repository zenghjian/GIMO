#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import tempfile
import zipfile
import random
from hdepic_dataset import HDEpicTrajectoryDataset

# Try to import rerun, handle gracefully if not available
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("Warning: rerun not available. Rerun visualization will be disabled.")
    print("Install with: pip install rerun-sdk")

# Try to import opencv, handle gracefully if not available
try:
    import cv2  # type: ignore
    OPENCV_AVAILABLE = True
    # Verify OpenCV has required attributes
    if not (hasattr(cv2, 'VideoCapture') and hasattr(cv2, 'CAP_PROP_FPS') and 
            hasattr(cv2, 'CAP_PROP_FRAME_COUNT') and hasattr(cv2, 'CAP_PROP_POS_FRAMES') and
            hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_BGR2RGB')):
        OPENCV_AVAILABLE = False
        print("Warning: OpenCV installation incomplete. Video playback will be disabled.")
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None
    print("Warning: opencv not available. Video playback will be disabled.")
    print("Install with: pip install opencv-python")

# Note: We're using VRS-based video loading instead of OpenCV for better performance

# Import Project Aria Tools for pointcloud and VRS video loading
try:
    import projectaria_tools.core.mps as aria_mps
    from projectaria_tools.core.mps.utils import filter_points_from_confidence
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId
    import tempfile
    import zipfile
    import glob
    ARIA_TOOLS_AVAILABLE = True
except ImportError:
    print("Warning: projectaria_tools not available")
    ARIA_TOOLS_AVAILABLE = False

def rotation_6d_to_quaternion(rotation_6d):
    """
    Convert 6D rotation representation to quaternion.
    
    Args:
        rotation_6d (torch.Tensor or np.ndarray): 6D rotation representation [a1, a2, b1, b2, c1, c2]
                                                  where a1,a2 and b1,b2 are two column vectors of rotation matrix
    
    Returns:
        np.ndarray: Quaternion [w, x, y, z]
    """
    if isinstance(rotation_6d, torch.Tensor):
        rotation_6d = rotation_6d.detach().cpu().numpy()
    
    # Handle zero/invalid rotations
    if np.allclose(rotation_6d, 0):
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    # Reshape to two 3D vectors
    a = rotation_6d[:3]  # First column vector
    b = rotation_6d[3:6]  # Second column vector
    
    # Normalize the first vector
    a = a / (np.linalg.norm(a) + 1e-8)
    
    # Gram-Schmidt process to orthogonalize
    b = b - np.dot(a, b) * a
    b = b / (np.linalg.norm(b) + 1e-8)
    
    # Third column vector (cross product)
    c = np.cross(a, b)
    
    # Construct rotation matrix
    R = np.column_stack([a, b, c])
    
    # Convert rotation matrix to quaternion
    # Using Shepperd's method for numerical stability
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])

def convert_displacement_to_absolute(positions, first_position=None):
    """Convert displacement trajectory to absolute positions."""
    if not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions)
        
    absolute_positions = torch.zeros_like(positions)
    
    if first_position is not None:
        absolute_positions[0] = first_position
    else:
        absolute_positions[0] = positions[0]  # First position stays the same
    
    # Accumulate displacements
    for i in range(1, len(positions)):
        absolute_positions[i, :3] = absolute_positions[i-1, :3] + positions[i, :3]
        absolute_positions[i, 3:] = positions[i, 3:]  # Keep orientation unchanged
    
    return absolute_positions

def load_sequence_pointcloud(seq_data, subsample_ratio=0.02):
    """Load pointcloud for a specific sequence using MPS with proper HD-EPIC structure"""
    if not ARIA_TOOLS_AVAILABLE:
        print(f"Cannot load pointcloud: ARIA_TOOLS_AVAILABLE={ARIA_TOOLS_AVAILABLE}")
        return None
        
    # Use the proper HD-EPIC SLAM directory structure
    participant = seq_data.participant
    hd_epic_base = "/storage/user/saroha/datasets/hd-epic-downloader/hd-epic/HD-EPIC"
    slam_base = f"{hd_epic_base}/SLAM-and-Gaze/{participant}/SLAM/multi"
    
    if not slam_base or not os.path.exists(slam_base):
        print(f"HD-EPIC SLAM directory not found: {slam_base}")
        return None
    
    print(f"Loading pointcloud from HD-EPIC SLAM directory: {slam_base}")
    
    try:
        # Find available zip files
        zip_files = glob.glob(os.path.join(slam_base, "*.zip"))
        if not zip_files:
            print(f"  No zip files found in {slam_base}")
            return None
        
        print(f"  Found {len(zip_files)} SLAM zip files")
        
        # Pick a medium-sized zip file for faster processing
        zip_info = [(f, os.path.getsize(f)) for f in zip_files]
        zip_info.sort(key=lambda x: x[1])  # Sort by size
        
        selected_zip = zip_info[len(zip_info)//3][0] if len(zip_info) > 2 else zip_info[0][0]
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
            point_cloud = filter_points_from_confidence(point_cloud, inverse_distance_std_threshold, distance_std_threshold)
            print(f"  Filtered to {len(point_cloud)} confident points")
            
            # Convert points to numpy array
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

def load_video_frames_fast_optimized(seq_data, max_frames=50, frame_skip=10, segment_frames=None):
    """Fast optimized video loading with segment support and frame reduction"""
    if not OPENCV_AVAILABLE or cv2 is None:
        print("OpenCV not available for video loading")
        return None
        
    video_path = seq_data.video_path
    if not video_path or not os.path.exists(video_path):
        print(f"Video path does not exist: {video_path}")
        return None
        
    print(f"Loading video frames (FAST MODE) from: {video_path}")
    
    try:
        cap = cv2.VideoCapture(video_path)  # type: ignore
        
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore
        fps = cap.get(cv2.CAP_PROP_FPS)  # type: ignore
        print(f"Video info: {total_frames} total frames, {fps:.2f} FPS")
        
        # Determine frame range to process
        if segment_frames:
            start_frame, end_frame = segment_frames
            start_frame = max(0, start_frame)
            end_frame = min(total_frames - 1, end_frame)
            print(f"Processing segment: frames {start_frame}-{end_frame}")
        else:
            start_frame = 0
            end_frame = min(total_frames - 1, max_frames * frame_skip)
            print(f"Processing first {end_frame - start_frame} frames")
        
        frames = []
        frame_idx = start_frame
        frames_loaded = 0
        
        # Fast frame loading with optimized seeking
        while frames_loaded < max_frames and frame_idx <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # type: ignore
            ret, frame = cap.read()
            
            if ret:
                # Convert and store every frame (no extra reduction)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
                frames.append((frame_rgb, frame_idx))
                frames_loaded += 1
                
                if frames_loaded % 50 == 0:
                    print(f"  Loaded {frames_loaded} frames so far...")
            
            frame_idx += frame_skip
            
        cap.release()
        print(f"FAST LOAD complete: {len(frames)} frames loaded (every {frame_skip}th frame)")
        return frames
        
    except Exception as e:
        print(f"Error in fast video loading: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_vrs_video_frames_efficiently(seq_data, max_frames=100, frame_skip=10):
    """Load video frames efficiently from VRS file using Project Aria Tools"""
    if not ARIA_TOOLS_AVAILABLE:
        print("Project Aria Tools not available, cannot load VRS video frames")
        return None
        
    vrs_path = seq_data.vrs_path
    if not vrs_path or not os.path.exists(vrs_path):
        print(f"VRS path does not exist: {vrs_path}")
        return None
        
    print(f"Loading VRS video frames from: {vrs_path}")
    
    try:
        # Open VRS file using the correct API
        vrs_provider = data_provider.create_vrs_data_provider(vrs_path)
        
        # Get RGB camera stream - try common RGB camera labels
        rgb_labels = ["camera-rgb", "rgb", "RGB"]
        rgb_stream_id = None
        
        for label in rgb_labels:
            try:
                stream_id = vrs_provider.get_stream_id_from_label(label)
                if stream_id and vrs_provider.check_stream_is_active(stream_id):
                    rgb_stream_id = stream_id
                    print(f"Found RGB stream using label '{label}': {rgb_stream_id}")
                    break
            except:
                continue
        
        if rgb_stream_id is None:
            # List all available streams and try to find an RGB-like stream
            print("No RGB stream found by label. Available streams:")
            try:
                all_streams = vrs_provider.get_all_streams()
                for stream in all_streams:
                    print(f"  Stream ID: {stream}")
                
                # Try to find RGB-like stream by examining stream IDs
                for stream in all_streams:
                    stream_str = str(stream)
                    if "rgb" in stream_str.lower() or "214" in stream_str or "1201" in stream_str:
                        rgb_stream_id = stream
                        print(f"Using RGB-like stream: {rgb_stream_id}")
                        break
                
                # Fallback to first stream
                if rgb_stream_id is None and all_streams:
                    rgb_stream_id = all_streams[0]
                    print(f"Using first available stream as fallback: {rgb_stream_id}")
                    
            except Exception as e:
                print(f"Error getting stream info: {e}")
                return None
        
        if rgb_stream_id is None:
            print("No suitable stream found in VRS file")
            return None
        
        # Get stream information
        try:
            num_data = vrs_provider.get_num_data(rgb_stream_id)
            print(f"VRS stream info: {num_data} data records in stream {rgb_stream_id}")
        except Exception as e:
            print(f"Error getting stream data count: {e}")
            return None
        
        frames = []
        frames_loaded = 0
        data_idx = 0
        error_count = 0
        
        print(f"Attempting to load frames from VRS...")
        
        while frames_loaded < max_frames and data_idx < num_data and error_count < 50:
            try:
                # Get image data record using the correct API
                image_data_record = vrs_provider.get_image_data_by_index(rgb_stream_id, data_idx)
                
                if image_data_record is not None:
                    # Try different ways to extract the image based on the API
                    image = None
                    
                    # Print debug info for first few frames
                    if data_idx < 30:
                        print(f"  Frame {data_idx}: image_data_record type: {type(image_data_record)}")
                        if hasattr(image_data_record, '__dict__'):
                            attrs = [attr for attr in dir(image_data_record) if not attr.startswith('_')]
                            print(f"  Frame {data_idx}: available attributes: {attrs[:10]}")  # First 10 attributes
                    
                    # Method 1: Check if it's a tuple/list with image data
                    if isinstance(image_data_record, (tuple, list)) and len(image_data_record) >= 2:
                        image_data = image_data_record[1]
                        if hasattr(image_data, 'to_numpy_array'):
                            image = image_data.to_numpy_array()
                            if data_idx < 10:
                                print(f"  Frame {data_idx}: Method 1 worked - extracted image shape: {image.shape}")
                    
                    # Method 2: Check if it has to_numpy_array directly
                    elif hasattr(image_data_record, 'to_numpy_array'):
                        image = image_data_record.to_numpy_array()
                        if data_idx < 10:
                            print(f"  Frame {data_idx}: Method 2 worked - extracted image shape: {image.shape}")
                    
                    # Method 3: Check if it's the image data directly
                    elif hasattr(image_data_record, 'image') and hasattr(image_data_record.image, 'to_numpy_array'):
                        image = image_data_record.image.to_numpy_array()
                        if data_idx < 10:
                            print(f"  Frame {data_idx}: Method 3 worked - extracted image shape: {image.shape}")
                    
                    # Method 4: Try accessing pixel_frame if it exists
                    elif hasattr(image_data_record, 'pixel_frame'):
                        pixel_frame = image_data_record.pixel_frame
                        if hasattr(pixel_frame, 'to_numpy_array'):
                            image = pixel_frame.to_numpy_array()
                            if data_idx < 10:
                                print(f"  Frame {data_idx}: Method 4 worked - extracted image shape: {image.shape}")
                    
                    # Method 5: Try accessing camera_data if it exists
                    elif hasattr(image_data_record, 'camera_data'):
                        camera_data = image_data_record.camera_data
                        if hasattr(camera_data, 'to_numpy_array'):
                            image = camera_data.to_numpy_array()
                            if data_idx < 10:
                                print(f"  Frame {data_idx}: Method 5 worked - extracted image shape: {image.shape}")
                    
                    if image is not None and image.size > 0:
                        # Convert to RGB if needed (VRS images might be in different formats)
                        if len(image.shape) == 3:
                            if image.shape[2] == 3:
                                # Already RGB, just ensure it's in the right format
                                image_rgb = image
                            elif image.shape[2] == 1:
                                # Grayscale, convert to RGB
                                image_rgb = np.stack([image[:,:,0]]*3, axis=2)
                            else:
                                # Unknown format, skip
                                data_idx += frame_skip
                                continue
                        else:
                            # Grayscale
                            image_rgb = np.stack([image]*3, axis=2)
                        
                        # Ensure values are in [0, 255] range
                        if image_rgb.dtype != np.uint8:
                            if image_rgb.max() <= 1.0:
                                image_rgb = (image_rgb * 255).astype(np.uint8)
                            else:
                                image_rgb = image_rgb.astype(np.uint8)
                        
                        frames.append((image_rgb, data_idx))
                        frames_loaded += 1
                        
                        if frames_loaded % 10 == 0:
                            print(f"  Loaded {frames_loaded} frames so far...")
                    else:
                        if data_idx < 10:
                            print(f"  Frame {data_idx}: No valid image data extracted")
                        
            except Exception as e:
                error_count += 1
                if data_idx < 10 or error_count % 10 == 0:
                    print(f"Error loading frame at index {data_idx}: {e}")
                
            data_idx += frame_skip
            
        print(f"Loaded {len(frames)} VRS video frames (every {frame_skip}th frame)")
        
        # If VRS loading failed, try fast OpenCV fallback
        if len(frames) == 0:
            print("VRS loading failed, trying fast OpenCV fallback...")
            return load_video_frames_fast_optimized(seq_data, max_frames, frame_skip)
        
        return frames
        
    except Exception as e:
        print(f"Error loading VRS video frames: {e}")
        import traceback
        traceback.print_exc()
        
        # Try fast OpenCV fallback
        print("VRS loading failed, trying fast OpenCV fallback...")
        return load_video_frames_fast_optimized(seq_data, max_frames, frame_skip)

def visualize_hdepic_in_rerun(max_trajectories=10):
    """
    Test the HD-EPIC trajectory dataset and visualize in Rerun
    
    Args:
        max_trajectories (int): Maximum number of trajectories to visualize (default: 10)
    """
    
    if not RERUN_AVAILABLE:
        print("Rerun is not available. Please install with: pip install rerun-sdk")
        return
    
    print("Testing HD-EPIC Trajectory Dataset...")
    
    # Check paths first
    base_path = "/storage/user/saroha/datasets/hd-epic-downloader/hd-epic/HD-EPIC"
    annotations_path = "/storage/user/saroha/datasets/hd-epic-annotations-main"
    
    print(f"Base path: {base_path}")
    print(f"Base path exists: {os.path.exists(base_path)}")
    print(f"Annotations path: {annotations_path}")
    print(f"Annotations path exists: {os.path.exists(annotations_path)}")
    
    if os.path.exists(base_path):
        base_contents = os.listdir(base_path)
        print(f"Base path contents: {base_contents}")
    
    # Check annotation files
    full_annotations_path = os.path.join(annotations_path, "scene-and-object-movements")
    assoc_path = os.path.join(full_annotations_path, "assoc_info.json")
    mask_path = os.path.join(full_annotations_path, "mask_info.json")
    print(f"Assoc file exists: {os.path.exists(assoc_path)}")
    print(f"Mask file exists: {os.path.exists(mask_path)}")
    
    # Initialize Rerun without spawning viewer (for saving to file)
    rr.init("HD-EPIC-Dataset-Test", spawn=False)
    
    try:
        # Create dataset with static and dynamic bbox loading
        dataset = HDEpicTrajectoryDataset(
            base_path="/storage/user/saroha/datasets/hd-epic-downloader/hd-epic/HD-EPIC/",
            annotations_path="/storage/user/saroha/datasets/hd-epic-annotations-main",
            trajectory_length=50,
            skip_frames=5,
            use_displacements=True,
            normalize_data=True,
            use_cache=True,  # 🔧 Disable cache to regenerate data with scene_bboxes
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
        
        print(f"\nDataset created successfully!")
        print(f"Total trajectories: {len(dataset)}")
        
        if len(dataset) == 0:
            print("No trajectories found! Check the dataset path and annotations.")
            return
        
        # Focus on one specific sequence for detailed visualization
        target_sequence = "P01-20240202-110250"  # Pick the first sequence
        
        # Find all trajectories from this sequence
        sequence_trajectories = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample.get('video_id', '') == target_sequence:
                sequence_trajectories.append((i, sample))
        
        print(f"\nFocusing on sequence: {target_sequence}")
        print(f"Found {len(sequence_trajectories)} trajectories in this sequence")
        
        if not sequence_trajectories:
            print("No trajectories found in target sequence")
            return
        
        # For the original dataset, we need to construct file paths manually
        participant = "P01"
        print(f"\nSequence details for {target_sequence}:")
        video_path = f"{base_path}/Videos/{participant}/{target_sequence}.mp4"
        vrs_path = f"{base_path}/VRS/{participant}/{target_sequence}_anonymized.vrs"
        slam_path = f"{base_path}/SLAM-and-Gaze/{participant}/SLAM"
        
        print(f"  Video path: {video_path}")
        print(f"  VRS path: {vrs_path}")
        print(f"  SLAM path: {slam_path}")
        print(f"  Video exists: {os.path.exists(video_path)}")
        print(f"  VRS exists: {os.path.exists(vrs_path)}")
        print(f"  SLAM exists: {os.path.exists(slam_path)}")
        
        # Create a simple sequence data object for compatibility
        class SimpleSeqData:
            def __init__(self):
                self.video_path = video_path if os.path.exists(video_path) else None
                self.vrs_path = vrs_path if os.path.exists(vrs_path) else None
                self.slam_path = slam_path if os.path.exists(slam_path) else None
                self.participant = participant
        
        seq_data = SimpleSeqData()
        
        # ================================================
        # Visualize scene pointcloud
        # ================================================
        # Load pointcloud for this sequence
        pointcloud = load_sequence_pointcloud(seq_data, subsample_ratio=0.1)
        if pointcloud is not None:
            print(f"Loaded pointcloud: {pointcloud.shape}")
            
            # Subsample pointcloud for visualization
            subsample_factor = max(1, len(pointcloud) // 50000)  # Max 50000 points
            pointcloud_viz = pointcloud[::subsample_factor]
            
            # Log pointcloud to Rerun
            rr.set_time("timeline", sequence=0)
            pointcloud_entity = f"scene/{target_sequence}/pointcloud"
            rr.log(
                pointcloud_entity,
                rr.Points3D(
                    positions=pointcloud_viz[:, :3],
                    colors=[[0.7, 0.7, 0.7]] * len(pointcloud_viz),
                    radii=[0.01] * len(pointcloud_viz),
                )
            )
            print(f"✅ Logged pointcloud to Rerun: {len(pointcloud_viz)} points at '{pointcloud_entity}'")
        
        # Load the full video without segment restrictions
        print(f"Loading full video without time restrictions")
        
        # Load video frames - full video
        video_frames = None
        
        # TODO: disable video frame for now
        # # Try MP4 first - load much more of the video
        # if seq_data.video_path:
        #     video_frames = load_video_frames_fast_optimized(seq_data, max_frames=50, frame_skip=10, segment_frames=None)
            
        # # Only try VRS if MP4 loading failed
        # if video_frames is None and seq_data.vrs_path:
        #     print("MP4 loading failed, trying VRS as backup...")
        #     video_frames = load_vrs_video_frames_efficiently(seq_data, max_frames=1000, frame_skip=10)
            
        # if video_frames:
        #     print(f"Loaded {len(video_frames)} video frames")
            
        #     # Create frame mapping for proper synchronization
        #     video_frame_mapping = {}
        #     for idx, (frame, actual_frame_idx) in enumerate(video_frames):
        #         video_frame_mapping[actual_frame_idx] = (idx, frame)
            
        #     print(f"Video frames available: {sorted(video_frame_mapping.keys())[:10]}...")
            
        #     # Log video frames to Rerun - simple sequential timeline
        #     video_entity = f"video/{target_sequence}"
        #     print(f"🎬 Logging {len(video_frames)} video frames to '{video_entity}'...")
            
        #     for idx, (frame, actual_frame_idx) in enumerate(video_frames):
        #         # Use simple sequential timeline for smooth playback
        #         rr.set_time_sequence("timeline", idx)
        #         rr.log(video_entity, rr.Image(frame))
                
        #         # Progress indicator
        #         if idx % 100 == 0:
        #             print(f"   Logged frame {idx+1}/{len(video_frames)}")
            
        #     print(f"✅ All video frames logged to '{video_entity}'")
        
        
        # ================================================
        # Visualize scene bounding boxes (Static and Dynamic)
        # ================================================
        
        # Generate random colors for different object types
        def get_bbox_color(object_name, bbox_type="dynamic", seed=None):
            """Get random color for bbox based on object name and type (consistent for same name)"""
            # Use object name as seed for consistent colors across runs
            if seed is None:
                seed = hash(f"{object_name.lower()}_{bbox_type}") % 2**32
            
            # Set seed for reproducible colors per object name
            random.seed(seed)
            
            # Different color schemes for static vs dynamic
            if bbox_type == "static":
                # Cooler colors for static objects (blues, grays, greens)
                base_colors = [
                    [0.2, 0.4, 0.8],  # Blue
                    [0.5, 0.5, 0.5],  # Gray
                    [0.2, 0.6, 0.3],  # Green
                    [0.4, 0.2, 0.6],  # Purple
                    [0.3, 0.5, 0.7],  # Light blue
                ]
            else:
                # Warmer colors for dynamic objects (reds, oranges, yellows)
                base_colors = [
                    [0.8, 0.2, 0.2],  # Red
                    [0.9, 0.5, 0.1],  # Orange
                    [0.8, 0.8, 0.2],  # Yellow
                    [0.8, 0.2, 0.6],  # Magenta
                    [0.6, 0.8, 0.2],  # Lime
                ]
            
            # Pick a base color and add some variation
            base_color = base_colors[hash(object_name) % len(base_colors)]
            color = [
                max(0.1, min(0.9, base_color[0] + random.uniform(-0.2, 0.2))),
                max(0.1, min(0.9, base_color[1] + random.uniform(-0.2, 0.2))),
                max(0.1, min(0.9, base_color[2] + random.uniform(-0.2, 0.2)))
            ]
            
            # Reset random seed to avoid affecting other random operations
            random.seed()
            
            return color
        
        # Extract scene bounding box data from the first sample
        if sequence_trajectories:
            first_sample = sequence_trajectories[0][1]  # Get first sample
            
            # Extract static bboxes
            static_bbox_info = first_sample.get('static_scene_bbox', torch.zeros((50, 12)))
            static_bbox_mask = first_sample.get('static_bbox_mask', torch.zeros(50))
            static_bbox_categories = first_sample.get('static_bbox_categories', ["unknown"] * 50)
            
            # Extract dynamic bboxes
            dynamic_bbox_info = first_sample.get('dynamic_scene_bbox', torch.zeros((50, 12)))
            dynamic_bbox_mask = first_sample.get('dynamic_bbox_mask', torch.zeros(50))
            dynamic_bbox_categories = first_sample.get('dynamic_bbox_categories', ["unknown"] * 50)
            
            # Extract valid static bboxes
            valid_static_indices = static_bbox_mask > 0
            valid_static_bboxes = static_bbox_info[valid_static_indices]
            valid_static_categories = [static_bbox_categories[i] for i in range(len(static_bbox_categories)) 
                                     if i < len(static_bbox_mask) and static_bbox_mask[i] > 0]
            
            # Extract valid dynamic bboxes
            valid_dynamic_indices = dynamic_bbox_mask > 0
            valid_dynamic_bboxes = dynamic_bbox_info[valid_dynamic_indices]
            valid_dynamic_categories = [dynamic_bbox_categories[i] for i in range(len(dynamic_bbox_categories)) 
                                      if i < len(dynamic_bbox_mask) and dynamic_bbox_mask[i] > 0]
            
            print(f"\n📦 Found {len(valid_static_bboxes)} static scene bounding boxes")
            print(f"📦 Found {len(valid_dynamic_bboxes)} dynamic scene bounding boxes")
            
            # ================================================
            # Visualize STATIC bounding boxes
            # ================================================
            if len(valid_static_bboxes) > 0:
                rr.set_time("timeline", sequence=0)  # Static display
                
                print(f"\n🏢 Visualizing {len(valid_static_bboxes)} static objects:")
                for i, (bbox_data, category) in enumerate(zip(valid_static_bboxes, valid_static_categories)):
                    # Extract bbox components: [center(3), size(3), rotation_6d(6)]
                    center = bbox_data[:3].numpy()
                    size = bbox_data[3:6].numpy()
                    rotation_6d = bbox_data[6:12].numpy()
                    
                    # Convert 6D rotation to quaternion using helper function
                    quaternion = rotation_6d_to_quaternion(rotation_6d)
                    
                    # Create bbox entity path for static objects
                    bbox_entity = f"scene/{target_sequence}/static_bboxes/{category}_{i}"
                    
                    # Get color for this static object type
                    bbox_color = get_bbox_color(category, "static")
                    
                    # Log 3D bounding box with rotation (wireframe style for static)
                    rr.log(
                        bbox_entity,
                        rr.Boxes3D(
                            centers=[center],
                            sizes=[size],
                            rotations=[quaternion],
                            colors=[bbox_color],
                            labels=[f"[S] {category}"],  # [S] prefix for static
                            show_labels=False
                        )
                    )
                    
                    print(f"  🏢 [STATIC] {category}: center={center}, size={size}")
                
                print(f"✅ Logged {len(valid_static_bboxes)} static bounding boxes to Rerun")
            else:
                print("⚠️  No valid static bounding boxes found")
            
            # ================================================
            # Visualize DYNAMIC bounding boxes
            # ================================================
            if len(valid_dynamic_bboxes) > 0:
                rr.set_time("timeline", sequence=0)  # Static display
                
                print(f"\n🎯 Visualizing {len(valid_dynamic_bboxes)} dynamic objects:")
                for i, (bbox_data, category) in enumerate(zip(valid_dynamic_bboxes, valid_dynamic_categories)):
                    # Extract bbox components: [center(3), size(3), rotation_6d(6)]
                    center = bbox_data[:3].numpy()
                    size = bbox_data[3:6].numpy()
                    rotation_6d = bbox_data[6:12].numpy()
                    
                    # Convert 6D rotation to quaternion using helper function
                    quaternion = rotation_6d_to_quaternion(rotation_6d)
                    
                    # Create bbox entity path for dynamic objects
                    bbox_entity = f"scene/{target_sequence}/dynamic_bboxes/{category}_{i}"
                    
                    # Get color for this dynamic object type
                    bbox_color = get_bbox_color(category, "dynamic")
                    
                    # Log 3D bounding box with rotation (solid style for dynamic)
                    rr.log(
                        bbox_entity,
                        rr.Boxes3D(
                            centers=[center],
                            sizes=[size],
                            rotations=[quaternion],
                            colors=[bbox_color],
                            labels=[f"[D] {category}"],  # [D] prefix for dynamic
                            show_labels=False
                        )
                    )
                    # INSERT_YOUR_CODE
                    # Log the bbox center as a 3D circle for visualization
                    rr.log(
                        f"{bbox_entity}/center",
                        rr.Points3D(
                            positions=[center],
                            colors=[bbox_color],
                            radii=[0.03],  # slightly larger for visibility
                            labels=[f"{category} center"],
                            show_labels=False
                        )
                    )
                    print(f"  🎯 [DYNAMIC] {category}: center={center}, size={size}")
                
                print(f"✅ Logged {len(valid_dynamic_bboxes)} dynamic bounding boxes to Rerun")
            else:
                print("⚠️  No valid dynamic bounding boxes found")
            
            # Summary
            total_bboxes = len(valid_static_bboxes) + len(valid_dynamic_bboxes)
            print(f"\n📊 Bounding Box Summary:")
            print(f"   Static objects: {len(valid_static_bboxes)}")
            print(f"   Dynamic objects: {len(valid_dynamic_bboxes)}")
            print(f"   Total objects: {total_bboxes}")
        else:
            print("⚠️  No sequence trajectories found for bbox visualization")
        
        # ================================================
        # Visualize trajectories
        # ================================================
        
        # Visualize only the first N trajectories from this sequence
        trajectories_to_show = sequence_trajectories[:max_trajectories]
        
        print(f"\n🎯 Limiting visualization to first {len(trajectories_to_show)} trajectories (out of {len(sequence_trajectories)} total)")
        
        for idx, (sample_idx, sample) in enumerate(trajectories_to_show):
            positions = sample['positions']
            video_id = sample.get('video_id', 'unknown')
            participant = sample.get('participant', 'unknown')
            object_name = sample.get('object_category', 'unknown')
            is_displacement = sample.get('is_displacement', False)
            
            print(f"\nTrajectory {idx}: {object_name}")
            
            # DEBUG: Print raw position data to understand the issue
            print(f"  Raw positions shape: {positions.shape}")
            print(f"  Raw positions dtype: {positions.dtype}")
            print(f"  Is displacement: {is_displacement}")
            print(f"  First few positions: {positions[:3]}")
            print(f"  Position range: min={positions.min().item():.4f}, max={positions.max().item():.4f}")
            
            # Convert to absolute positions if needed
            if is_displacement and 'first_position' in sample:
                first_pos = sample['first_position']
                positions_abs = convert_displacement_to_absolute(positions, first_pos)
                print(f"  Converted from displacement with first_pos: {first_pos}")
            elif is_displacement:
                positions_abs = convert_displacement_to_absolute(positions)
                print(f"  Converted from displacement without first_pos")
            else:
                positions_abs = positions
                print(f"  Using absolute positions directly")
            
            # DEBUG: Print converted position data
            print(f"  Final positions shape: {positions_abs.shape}")
            print(f"  Final first few positions: {positions_abs[:3]}")
            print(f"  Final position range: min={positions_abs.min().item():.4f}, max={positions_abs.max().item():.4f}")
            
            # Calculate motion
            total_motion = 0
            if len(positions_abs) > 1:
                for j in range(1, len(positions_abs)):
                    motion = (positions_abs[j, :3] - positions_abs[j-1, :3]).norm().item()
                    total_motion += motion
            print(f"  Total motion: {total_motion:.4f}m")
            
            # Log trajectory to Rerun
            positions_np = positions_abs.detach().cpu().numpy()
            
            # Use different colors for different objects - cycle through colors for many trajectories
            colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
                [1.0, 0.5, 0.0],  # Orange
                [0.5, 0.0, 1.0],  # Purple
                [0.0, 0.5, 1.0],  # Light Blue
                [1.0, 0.0, 0.5],  # Pink
                [0.5, 1.0, 0.0],  # Lime
                [1.0, 0.5, 0.5],  # Light Red
                [0.5, 0.5, 1.0],  # Light Blue
                [1.0, 1.0, 0.5],  # Light Yellow
                [0.5, 1.0, 1.0],  # Light Cyan
                [1.0, 0.5, 1.0],  # Light Magenta
                [0.8, 0.4, 0.2],  # Brown
                [0.2, 0.8, 0.4],  # Green-ish
                [0.4, 0.2, 0.8],  # Purple-ish
                [0.8, 0.8, 0.2],  # Yellow-green
            ] 
            color = colors[idx % len(colors)]
            
            # Log trajectory path
            trajectory_entity = f"trajectories/{target_sequence}/{object_name}_path"
            rr.log(
                trajectory_entity,
                rr.LineStrips3D(
                    strips=[positions_np[:, :3]],
                    colors=[color],
                )
            )
            print(f"  ✅ Logged trajectory path to '{trajectory_entity}'")
            
            # Simplified trajectory animation - spread across the video timeline
            total_video_frames = len(video_frames) if video_frames else 500
            trajectory_spread = total_video_frames // len(positions_np) if len(positions_np) > 0 else 10
            
            for traj_idx, pos in enumerate(positions_np):
                # Spread trajectory across the video timeline
                timeline_frame = traj_idx * trajectory_spread
                rr.set_time("timeline", sequence=timeline_frame)
                
                # Show current trajectory point
                rr.log(
                    f"trajectories/{target_sequence}/{object_name}_points",
                    rr.Points3D(
                        positions=[pos[:3]],
                        colors=[color],
                        radii=[0.02],
                    )
                )
                
                # Show progressive trajectory path
                if traj_idx > 0:
                    path_so_far = positions_np[:traj_idx+1, :3]
                    rr.log(
                        f"trajectories/{target_sequence}/{object_name}_path_progress",
                        rr.LineStrips3D(
                            strips=[path_so_far],
                            colors=[color],
                            radii=[0.005],
                        )
                    )
                    
            print(f"  Trajectory spread across timeline: 0 to {(len(positions_np)-1) * trajectory_spread}")
            
            # Log trajectory metadata
            rr.set_time("timeline", sequence=0)
            rr.log(
                f"trajectories/{target_sequence}/{object_name}/info",
                rr.TextLog(
                    f"Object: {object_name}\n"
                    f"Motion: {total_motion:.4f}m\n"
                    f"Points: {len(positions_np)}\n"
                    f"Type: {'displacement' if is_displacement else 'absolute'}"
                )
            )
        
        print(f"\n✅ HD-EPIC trajectory visualization completed!")
        print(f"   - Focused on sequence: {target_sequence}")
        print(f"   - Visualized {len(trajectories_to_show)} trajectories (limited from {len(sequence_trajectories)} total)")
        print(f"   - Pointcloud: {'✓' if pointcloud is not None else '✗'}")
        print(f"   - Video frames: {'✓' if video_frames else '✗'}")
        
        # Updated bbox summary
        static_count = len(valid_static_bboxes) if 'valid_static_bboxes' in locals() else 0
        dynamic_count = len(valid_dynamic_bboxes) if 'valid_dynamic_bboxes' in locals() else 0
        total_bbox_count = static_count + dynamic_count
        
        print(f"   - Static bounding boxes: {'✓' if static_count > 0 else '✗'} ({static_count} boxes)")
        print(f"   - Dynamic bounding boxes: {'✓' if dynamic_count > 0 else '✗'} ({dynamic_count} boxes)")
        print(f"   - Total bounding boxes: {total_bbox_count}")
        print(f"   - Total dataset: {len(dataset)} trajectories")
        
        # # Ensure all data is flushed before saving
        # print(f"\n🔄 Flushing Rerun data before saving...")
        
        # # Force flush any pending operations
        # rr.set_time_sequence("timeline", 0)
        # rr.log("flush_marker", rr.TextLog("Data flush marker"))
        
        # Save Rerun data to file
        rerun_output_dir = "./hdepic"
        os.makedirs(rerun_output_dir, exist_ok=True)
        output_file = os.path.join(rerun_output_dir, f"HD-EPIC-Dataset-Test_{target_sequence}.rrd")
        
        print(f"💾 Attempting to save Rerun data to: {output_file}")
        try:
            rr.save(output_file)
            
            # Check if file was actually created and has content
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✅ Rerun data saved successfully!")
                print(f"   File: {output_file}")
                print(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                print(f"   Absolute path: {os.path.abspath(output_file)}")
                
                if file_size < 1000:  # Less than 1KB is probably empty
                    print(f"⚠️  WARNING: File size is very small ({file_size} bytes), may be empty!")
            else:
                print(f"❌ File was not created: {output_file}")
                
        except Exception as e:
            print(f"❌ Failed to save Rerun data: {e}")
            import traceback
            traceback.print_exc()
        
            
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Visualize only the first 10 trajectories for faster processing
    # You can change this number to visualize more or fewer trajectories
    MAX_TRAJECTORIES = 3
    
    print(f"🎯 Starting HD-EPIC visualization with max {MAX_TRAJECTORIES} trajectories...")
    visualize_hdepic_in_rerun(max_trajectories=MAX_TRAJECTORIES) 