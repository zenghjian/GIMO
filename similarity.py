#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import KDTree
import time
from functools import partial
from torch.utils.data.dataloader import default_collate

# Import modules from the project
from config.adt_config import ADTObjectMotionConfig
from dataset.gimo_multi_sequence_dataset import GIMOMultiSequenceDataset
from utils.visualization import visualize_full_trajectory

# Global variable to store script arguments so other functions can access them
script_args = None

def setup_script_parser():
    """Sets up the argument parser for this script."""
    parser = argparse.ArgumentParser(description="Verify trajectory similarity between training and validation sets.", add_help=False) # add_help=False to avoid conflict with ADTConfig's -h
    parser.add_argument('--config', type=str, default=None, help='Configuration file path (specific to similarity.py)')
    parser.add_argument('--train_sequences', type=str, default='train_sequences.txt', help='Path to the training sequences file')
    parser.add_argument('--val_sequences', type=str, default='val_sequences.txt', help='Path to the validation sequences file')
    parser.add_argument('--global_cache_dir', type=str, default=None, help='Global cache directory, same as used during training')
    parser.add_argument('--output_dir', type=str, default='similarity_results', help='Directory for output results')
    parser.add_argument('--top-k', type=int, default=3, help='Number of most similar training trajectories to show for each validation trajectory')
    parser.add_argument('--similarity-threshold', type=float, default=0.3, help='Threshold to mark highly similar trajectories')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loading (specific to similarity.py)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading (specific to similarity.py)')
    return parser

def collate_fn(batch, dataset, num_sample_points):
    """
    Custom collate function to handle trajectory data and point clouds.
    """
    point_cloud_batch_list = []
    items_for_default_collate = []

    # Define keys needed for extract_trajectory_features
    # and also keys that GIMOMultiSequenceDataset.__getitem__ returns and default_collate can safely handle
    # Note: 'dataset_idx' is added by GIMOMultiSequenceDataset and also used in extract_trajectory_features
    # Other fields (like segment_idx, object_id) in GIMOMultiSequenceDataset.__getitem__ should also be safe.
    # String fields (object_name, sequence_name, object_category, sequence_path) will be returned as lists by default_collate.
    
    # These are keys explicitly used in extract_trajectory_features
    keys_needed_downstream = [
        'full_poses', 'full_attention_mask', 
        'object_name', 'sequence_name', 'segment_idx', 
        'object_category', 'dataset_idx'
    ]
    
    # Consider other possible keys that could be safe to include in default_collate
    # (if downstream code indirectly needs them, or to maintain sample structure integrity)
    # For example: 'normalization', 'bbox_corners', 'object_id', 'object_category_id', 'original_category_id'
    # 'first_position' is still a potential issue if it mixes None and Tensor.
    # We'll only include keys_needed_downstream to ensure safety.

    for item in batch:
        # 1. Process point cloud (similar to previous logic)
        dataset_idx_for_pc = item.get('dataset_idx', 0)
        point_cloud = None
        
        # Try to get from 'trajectory_specific_pointcloud'
        if 'trajectory_specific_pointcloud' in item and item['trajectory_specific_pointcloud'] is not None:
            point_cloud_src = item['trajectory_specific_pointcloud']
            if isinstance(point_cloud_src, np.ndarray):
                point_cloud = torch.from_numpy(point_cloud_src).float()
            elif isinstance(point_cloud_src, torch.Tensor):
                point_cloud = point_cloud_src.float()
        
        # If above fails or not provided, fall back to get_scene_pointcloud
        if point_cloud is None and hasattr(dataset, 'get_scene_pointcloud'):
            point_cloud_src = dataset.get_scene_pointcloud(dataset_idx_for_pc)
            if point_cloud_src is None:
                point_cloud = torch.zeros((num_sample_points, 3), dtype=torch.float32)
            elif isinstance(point_cloud_src, np.ndarray):
                point_cloud = torch.from_numpy(point_cloud_src).float()
            elif isinstance(point_cloud_src, torch.Tensor):
                point_cloud = point_cloud_src.float()
            else: # Final fallback
                point_cloud = torch.zeros((num_sample_points, 3), dtype=torch.float32)
        
        # If all methods fail, ensure point_cloud is a zero tensor
        if point_cloud is None:
             point_cloud = torch.zeros((num_sample_points, 3), dtype=torch.float32)

        # Sample point cloud
        if point_cloud.shape[0] == 0: # Empty point cloud case
            sampled_pc = torch.zeros((num_sample_points, 3), dtype=torch.float32)
        elif point_cloud.shape[0] >= num_sample_points:
            indices = torch.randperm(point_cloud.shape[0])[:num_sample_points]
            sampled_pc = point_cloud[indices]
        else: # Insufficient points, with replacement sampling
            indices = torch.randint(0, point_cloud.shape[0], (num_sample_points,))
            sampled_pc = point_cloud[indices]
        point_cloud_batch_list.append(sampled_pc)

        # 2. Prepare items to pass to default_collate
        current_item_for_collate = {}
        for key in keys_needed_downstream: # Only select needed and safe keys
            if key in item:
                current_item_for_collate[key] = item[key]
            # else: If key doesn't exist, default_collate might raise an error,
            # but GIMOMultiSequenceDataset.__getitem__ should ensure these keys exist.
        items_for_default_collate.append(current_item_for_collate)

    # 3. Call default_collate on cleaned items
    collated_batch = default_collate(items_for_default_collate)
    
    # 4. Add manually processed point clouds
    if point_cloud_batch_list:
        collated_batch['point_cloud'] = torch.stack(point_cloud_batch_list, dim=0)
    else: # If batch is empty or point cloud processing fails
        # Create a correctly shaped but empty tensor, or handle as needed
        # Assuming batch size can be obtained via len(items_for_default_collate) if it's not empty
        batch_size_for_pc = len(items_for_default_collate) if items_for_default_collate else 0
        collated_batch['point_cloud'] = torch.zeros((batch_size_for_pc, num_sample_points, 3), dtype=torch.float32)
        
    return collated_batch

def load_dataset(config_obj, sequence_file, is_train=True):
    """
    Loads the dataset.
    """
    print(f"Loading {'training' if is_train else 'validation'} dataset...")
    
    with open(sequence_file, 'r') as f:
        sequence_paths = [line.strip() for line in f if line.strip()]
    
    # Ensure global_cache_dir is from script_args (if provided) or adt_config
    cache_dir_to_use = script_args.global_cache_dir
    if cache_dir_to_use is None and hasattr(config_obj, 'global_cache_dir'):
        cache_dir_to_use = config_obj.global_cache_dir
    
    dataset = GIMOMultiSequenceDataset(
        sequence_paths=sequence_paths,
        config=config_obj,
        cache_dir=cache_dir_to_use
    )
    
    print(f"Loaded {len(dataset)} trajectory samples from {sequence_file}")
    return dataset

def extract_trajectory_features(dataset, device, num_sample_points_for_collate):
    """
    Extracts features and metadata from all trajectories in the dataset.
    """
    print("Extracting trajectory features...")
    
    features = []
    metadata = []
    
    custom_collate = partial(collate_fn, dataset=dataset, num_sample_points=num_sample_points_for_collate)
    
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=script_args.batch_size, 
        shuffle=False, 
        num_workers=script_args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    for batch in tqdm(loader, desc="Extracting features"):
        full_poses = batch['full_poses'].to(device)
        attention_mask = batch['full_attention_mask'].to(device)
        
        object_names_list = batch.get('object_name', [f"unknown_{i}" for i in range(full_poses.shape[0])])
        sequence_names_list = batch.get('sequence_name', ['unknown'] * full_poses.shape[0])
        segment_indices_tensor = batch.get('segment_idx') # Tensor or None
        object_categories_list = batch.get('object_category', ['unknown'] * full_poses.shape[0])
        dataset_indices_tensor = batch.get('dataset_idx') # Tensor or None

        for i in range(full_poses.shape[0]):
            poses = full_poses[i]
            mask = attention_mask[i]
            positions = poses[:, :3]
            valid_indices = torch.where(mask > 0.5)[0]

            if len(valid_indices) > 0:
                valid_positions = positions[valid_indices]
                features.append(valid_positions.cpu().numpy())
                
                seg_idx_val = None
                if segment_indices_tensor is not None and i < len(segment_indices_tensor):
                    seg_idx_item = segment_indices_tensor[i].item()
                    if seg_idx_item != -1: # -1 usually indicates missing
                        seg_idx_val = seg_idx_item
                
                dataset_idx_val = 0
                if dataset_indices_tensor is not None and i < len(dataset_indices_tensor):
                     dataset_idx_val = dataset_indices_tensor[i].item()

                meta = {
                    'object_name': object_names_list[i],
                    'sequence_name': sequence_names_list[i],
                    'segment_idx': seg_idx_val,
                    'object_category': object_categories_list[i],
                    'dataset_idx': dataset_idx_val
                }
                metadata.append(meta)
    
    print(f"Extracted features for {len(features)} trajectories")
    return features, metadata

def compute_trajectory_distance(traj1, traj2, method='average_l2'):
    """
    Computes the distance/similarity between two trajectories.
    
    Args:
        traj1, traj2: numpy arrays of shape [n_points, 3] and [m_points, 3]
        method: Method for distance calculation.
    
    Returns:
        distance: Distance between trajectories, smaller means more similar.
    """
    if len(traj1) == 0 or len(traj2) == 0:
        return float('inf')

    if method == 'average_l2':
        dist1 = float('inf')
        if len(traj1) >= len(traj2) and len(traj2) > 0: # traj1 is longer or equal
            indices = np.linspace(0, len(traj1)-1, len(traj2), dtype=int)
            sampled_traj1 = traj1[indices]
            dist1 = np.mean(np.linalg.norm(sampled_traj1 - traj2, axis=1))
        elif len(traj1) < len(traj2) and len(traj1) > 0: # traj1 is shorter
            indices = np.linspace(0, len(traj2)-1, len(traj1), dtype=int)
            sampled_traj2_for_traj1 = traj2[indices]
            dist1 = np.mean(np.linalg.norm(traj1 - sampled_traj2_for_traj1, axis=1))
            
        dist2 = float('inf')
        if len(traj2) >= len(traj1) and len(traj1) > 0: # traj2 is longer or equal
            indices = np.linspace(0, len(traj2)-1, len(traj1), dtype=int)
            sampled_traj2 = traj2[indices]
            dist2 = np.mean(np.linalg.norm(traj1 - sampled_traj2, axis=1))
        elif len(traj2) < len(traj1) and len(traj2) > 0: # traj2 is shorter
            indices = np.linspace(0, len(traj1)-1, len(traj2), dtype=int)
            sampled_traj1_for_traj2 = traj1[indices]
            dist2 = np.mean(np.linalg.norm(sampled_traj1_for_traj2 - traj2, axis=1))
            
        return min(dist1, dist2) if min(dist1, dist2) != float('inf') else float('inf')
    
    elif method == 'dtw':
        from scipy.spatial.distance import cdist
        
        dist_matrix = cdist(traj1, traj2)
        n, m = len(traj1), len(traj2)
        dtw_matrix = np.full((n+1, m+1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = dist_matrix[i-1, j-1]
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
                
        return dtw_matrix[n, m] / (n + m) if (n + m) > 0 else 0
    
    else:
        raise ValueError(f"Unknown distance calculation method: {method}")

def find_nearest_neighbors(val_features, val_metadata, train_features, train_metadata, top_k=3, method='average_l2'):
    """
    Finds the most similar training trajectories for each validation trajectory.
    """
    print("Finding nearest neighbors...")
    results = []
    
    for i, val_feature in enumerate(tqdm(val_features, desc="Finding nearest neighbors")):
        if len(val_feature) == 0: continue
        distances = []
        for j, train_feature in enumerate(train_features):
            if len(train_feature) == 0:
                distances.append((float('inf'), j))
                continue
            dist = compute_trajectory_distance(val_feature, train_feature, method=method)
            distances.append((dist, j))
        
        distances.sort()
        valid_distances = [(d, idx) for d, idx in distances if d != float('inf')]
        
        top_k_indices = [idx for _, idx in valid_distances[:top_k]]
        top_k_distances = [dist for dist, _ in valid_distances[:top_k]]
        
        if not top_k_indices: 
            continue

        results.append({
            'val_index': i, 'val_metadata': val_metadata[i],
            'train_indices': top_k_indices,
            'train_metadata': [train_metadata[idx] for idx in top_k_indices],
            'distances': top_k_distances
        })
    return results

def visualize_trajectory_pair(val_feature, val_metadata, train_feature, train_metadata, 
                            distance, save_path=None, point_clouds=None):
    """
    Visualizes a pair of validation and training trajectories.
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), subplot_kw={'projection': '3d'})
    
    val_obj_name = val_metadata['object_name']
    val_seq_name = val_metadata['sequence_name']
    val_seg_idx = val_metadata['segment_idx']
    train_obj_name = train_metadata['object_name']
    train_seq_name = train_metadata['sequence_name']
    train_seg_idx = train_metadata['segment_idx']
    val_seg_str = f"seg{val_seg_idx}" if val_seg_idx is not None else "segNA"
    train_seg_str = f"seg{train_seg_idx}" if train_seg_idx is not None else "segNA"
    
    for ax_idx, (ax, feature, title_prefix, pc_data) in enumerate(zip(
        axes, 
        [val_feature, train_feature], 
        [f"Validation: {val_obj_name} ({val_seq_name}, {val_seg_str})", f"Training: {train_obj_name} ({train_seq_name}, {train_seg_str})"],
        point_clouds if point_clouds is not None else [None, None]
    )):
        if pc_data is not None:
            pc = pc_data.detach().cpu().numpy() if isinstance(pc_data, torch.Tensor) else pc_data
            if len(pc) > 0:
                max_points = 5000
                if len(pc) > max_points:
                    indices = np.random.choice(len(pc), max_points, replace=False)
                    pc = pc[indices]
                ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='gray', s=0.5, alpha=0.1)

        if len(feature) > 0:
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature)))
            for i in range(len(feature) - 1):
                ax.plot(feature[i:i+2, 0], feature[i:i+2, 1], feature[i:i+2, 2], color=colors[i], linewidth=2)
            ax.scatter(feature[:, 0], feature[:, 1], feature[:, 2], c=colors, s=30)
            ax.scatter(feature[0, 0], feature[0, 1], feature[0, 2], c='lime', marker='o', s=100, label='Start', edgecolors='black')
            ax.scatter(feature[-1, 0], feature[-1, 1], feature[-1, 2], c='red', marker='X', s=100, label='End', edgecolors='black')
        
        ax.set_title(title_prefix)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend()

    plt.suptitle(f'Trajectory Similarity Comparison (Distance = {distance:.4f})', fontsize=16)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

def generate_report(results, output_dir, similarity_threshold):
    """
    Generates a similarity analysis report.
    """
    print("Generating analysis report...")
    report_path = os.path.join(output_dir, 'similarity_report.txt')
    hist_path = os.path.join(output_dir, 'similarity_histogram.png')
    
    all_distances = [d for res in results for d in res['distances'] if d != float('inf')]
    if not all_distances:
        print("Warning: No valid distance values found, cannot generate report statistics.")
        with open(report_path, 'w') as f: f.write("Error: No valid distance values for analysis.\n")
        return

    min_dist, max_dist = min(all_distances), max(all_distances)
    mean_dist, median_dist, std_dist = np.mean(all_distances), np.median(all_distances), np.std(all_distances)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_distances, bins=50, alpha=0.7, range=(min_dist, max_dist if max_dist > min_dist else min_dist + 1))
    plt.axvline(similarity_threshold, color='r', linestyle='--', label=f'Threshold = {similarity_threshold:.2f}')
    plt.xlabel('Distance (smaller is more similar)'); plt.ylabel('Frequency'); plt.title('Trajectory Similarity Distribution')
    plt.legend(); plt.grid(True, alpha=0.3); plt.savefig(hist_path); plt.close()
    
    high_similarity_count = sum(1 for d in all_distances if d < similarity_threshold)
    
    with open(report_path, 'w') as f:
        f.write(f"Trajectory Similarity Analysis Report\n==================================\n\n")
        f.write(f"Analyzed similarity for {len(results)} validation set trajectories against training set trajectories.\n\n")
        f.write(f"Statistics:\n  Min Distance: {min_dist:.6f}\n  Max Distance: {max_dist:.6f}\n  Mean Distance: {mean_dist:.6f}\n")
        f.write(f"  Median Distance: {median_dist:.6f}\n  Std Dev Distance: {std_dist:.6f}\n\n")
        f.write(f"Highly Similar Trajectory Pairs:\n  Threshold: {similarity_threshold}\n  Count: {high_similarity_count}\n\n")
        
        flat_results = []
        for res_item in results:
            val_meta = res_item['val_metadata']
            for train_meta, dist in zip(res_item['train_metadata'], res_item['distances']):
                if dist == float('inf'): continue
                flat_results.append({'val_meta': val_meta, 'train_meta': train_meta, 'distance': dist})
        
        flat_results.sort(key=lambda x: x['distance'])
        f.write("Top 10 Most Similar Trajectory Pairs:\n")
        for i, data in enumerate(flat_results[:10]):
            vm, tm = data['val_meta'], data['train_meta']
            f.write(f"  {i+1}. Distance: {data['distance']:.6f}\n")
            f.write(f"     Validation: {vm['object_name']} (Seq: {vm['sequence_name']}, Seg: {vm['segment_idx']})\n")
            f.write(f"     Training:   {tm['object_name']} (Seq: {tm['sequence_name']}, Seg: {tm['segment_idx']})\n\n")
        
        f.write("\nPotential Data Leakage Analysis:\n")
        if high_similarity_count > 0:
            f.write(f"Found {high_similarity_count} highly similar trajectory pairs, which might indicate data leakage.\n")
            f.write("It is recommended to review these pairs and the dataset splitting process.\n")
        else:
            f.write("No highly similar trajectory pairs found based on the current threshold. Risk of data leakage appears low.\n")
    
    print(f"Report saved to: {report_path}\nSimilarity histogram saved to: {hist_path}")

def main():
    global script_args 
    parser = setup_script_parser()
    script_args_namespace, unknown_argv = parser.parse_known_args()
    script_args = script_args_namespace
    os.makedirs(script_args.output_dir, exist_ok=True)
    
    original_sys_argv = sys.argv
    sys.argv = [original_sys_argv[0]] + unknown_argv 
    print(f"Arguments passed to ADTObjectMotionConfig: {sys.argv}")
    adt_config_loader = ADTObjectMotionConfig()
    adt_config = adt_config_loader.get_configs()
    sys.argv = original_sys_argv

    if script_args.global_cache_dir is not None:
        adt_config.global_cache_dir = script_args.global_cache_dir
    elif not hasattr(adt_config, 'global_cache_dir') or adt_config.global_cache_dir is None:
        print("Warning: global_cache_dir not specified. GIMOMultiSequenceDataset might use a default cache location.")
        adt_config.global_cache_dir = './trajectory_cache' 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_points_for_collate = getattr(adt_config, 'sample_points', 1000)

    train_dataset = load_dataset(adt_config, script_args.train_sequences, is_train=True)
    val_dataset = load_dataset(adt_config, script_args.val_sequences, is_train=False)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: One or both datasets are empty. Please check sequence files and cache.")
        return

    train_features, train_metadata = extract_trajectory_features(train_dataset, device, num_points_for_collate)
    val_features, val_metadata = extract_trajectory_features(val_dataset, device, num_points_for_collate)
    
    if not train_features or not val_features:
        print("Error: Failed to extract features from one or both datasets.")
        return

    results = find_nearest_neighbors(
        val_features, val_metadata, 
        train_features, train_metadata, 
        top_k=script_args.top_k
    )
    
    print("Visualizing trajectory pairs...")
    visualization_dir = os.path.join(script_args.output_dir, 'visualizations')
    os.makedirs(visualization_dir, exist_ok=True)
    
    for result_item in tqdm(results, desc="Generating visualizations"):
        val_original_idx = result_item['val_index']
        val_meta = result_item['val_metadata']
        val_pc_data = None 

        for i, (train_original_idx, train_meta, dist) in enumerate(zip(
            result_item['train_indices'], 
            result_item['train_metadata'], 
            result_item['distances'] 
        )):
            if dist == float('inf'): continue # Skip if distance is infinity

            val_obj_name = val_meta['object_name'].replace('/', '_').replace(' ', '_')
            val_seq_name = val_meta['sequence_name'].replace('/', '_').replace(' ', '_')
            val_seg_idx = val_meta['segment_idx']
            val_seg_str = f"s{val_seg_idx}" if val_seg_idx is not None else "sNA"
            
            train_obj_name = train_meta['object_name'].replace('/', '_').replace(' ', '_')
            train_seq_name = train_meta['sequence_name'].replace('/', '_').replace(' ', '_')
            train_seg_idx = train_meta['segment_idx']
            train_seg_str = f"s{train_seg_idx}" if train_seg_idx is not None else "sNA"
            
            filename_core = f"val_{val_obj_name[:15]}_{val_seq_name[:30]}_{val_seg_str}_trn_{train_obj_name[:15]}_{train_seq_name[:30]}_{train_seg_str}_sim_{dist:.3f}"
            filename = f"{filename_core}.png"
            filepath = os.path.join(visualization_dir, filename)
            
            train_pc_data = None
            point_clouds_for_vis = [val_pc_data, train_pc_data]

            visualize_trajectory_pair(
                val_features[val_original_idx], val_meta, 
                train_features[train_original_idx], train_meta, 
                dist, save_path=filepath, point_clouds=point_clouds_for_vis
            )
            
            if dist < script_args.similarity_threshold:
                high_similarity_dir = os.path.join(script_args.output_dir, 'high_similarity_visuals')
                os.makedirs(high_similarity_dir, exist_ok=True)
                high_sim_filepath = os.path.join(high_similarity_dir, filename)
                if os.path.exists(filepath):
                    import shutil
                    shutil.copy(filepath, high_sim_filepath)
    
    generate_report(results, script_args.output_dir, script_args.similarity_threshold)
    print(f"Analysis complete!\nResults saved to directory: {script_args.output_dir}")

if __name__ == "__main__":
    main()