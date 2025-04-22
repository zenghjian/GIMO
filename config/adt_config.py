from argparse import ArgumentParser

class ADTObjectMotionConfig(ArgumentParser):
    def __init__(self):
        super().__init__()

        # === Input Data Configuration ===
        self.input_configs = self.add_argument_group('Input')
        self.input_configs.add_argument('--batch_size', default=2, type=int, help='Batch size for training/evaluation')
        self.input_configs.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading')

        # === Trajectory & Dataset Configuration ===
        self.traj_dataset_configs = self.add_argument_group('Trajectory & Dataset')
        self.traj_dataset_configs.add_argument('--adt_dataroot', default='./data', type=str, help='Path to ADT sequence or directory containing sequences/split files')
        self.traj_dataset_configs.add_argument('--train_split_file', default='./splits/train_sequences.txt', type=str, help='Path to train split file')
        self.traj_dataset_configs.add_argument('--test_split_file', default='./splits/test_sequences.txt', type=str, help='Path to test split file')
        self.traj_dataset_configs.add_argument('--trajectory_length', default=100, type=int, help='Total length of trajectory segments extracted (before split)')
        self.traj_dataset_configs.add_argument('--history_fraction', default=0.6, type=float, help='Fraction of trajectory_length to use for history (e.g., 0.6 for 3/5 ratio)')
        self.traj_dataset_configs.add_argument('--object_motion_dim', default=3, type=int, help='Dimension of object trajectory points (e.g., 3 for XYZ)')
        self.traj_dataset_configs.add_argument('--point_cloud_dim', default=3, type=int, help='Dimension of input point cloud points (e.g., 3 for XYZ)')
        self.traj_dataset_configs.add_argument('--skip_frames', default=5, type=int, help='Frames to skip when extracting trajectory points from dataset')
        self.traj_dataset_configs.add_argument('--use_displacements', default=False, action='store_true', help='Use relative displacements instead of absolute positions')
        self.traj_dataset_configs.add_argument('--load_pointcloud', default=True, type=bool, help='Whether to load point cloud data from sequences')
        self.traj_dataset_configs.add_argument('--pointcloud_subsample', default=10, type=int, help='Subsample factor for point cloud (higher means fewer points)')
        self.traj_dataset_configs.add_argument('--min_motion_threshold', default=1.0, type=float, help='Minimum motion threshold in meters for trajectory filtering')
        self.traj_dataset_configs.add_argument('--min_motion_percentile', default=0.0, type=float, help='Filter trajectories below this percentile of motion')
        self.traj_dataset_configs.add_argument('--use_cache', default=True, type=bool, help='Whether to use caching for trajectories')
        self.traj_dataset_configs.add_argument('--detect_motion_segments', default=True, type=bool, help='Whether to detect and extract active motion segments')
        self.traj_dataset_configs.add_argument('--motion_velocity_threshold', default=0.05, type=float, help='Threshold in m/s for detecting active motion')
        self.traj_dataset_configs.add_argument('--min_segment_frames', default=5, type=int, help='Minimum number of frames for a valid motion segment')
        self.traj_dataset_configs.add_argument('--max_stationary_frames', default=3, type=int, help='Maximum consecutive stationary frames allowed in a motion segment')
        self.traj_dataset_configs.add_argument('--normalize_data', default=True, type=bool, help='Whether to normalize trajectory data using scene bounds')

        # === Scene/Point Cloud Configuration ===
        self.scene_configs = self.add_argument_group('Scene Encoder')
        self.scene_configs.add_argument('--scene_feats_dim', default=256, type=int, help='Output dimension of the PointNet scene encoder')
        self.scene_configs.add_argument('--sample_points', default=20000, type=int, help='Number of points to sample from the scene point cloud') # Reduced default for potentially large clouds
        # self.scene_configs.add_argument('--pointnet_chkpoints', default='pretrained/point.model', type=str, help='Path to pretrained PointNet weights') # Consider adding back later

        # === Motion Pathway Configuration (Based on GIMO) ===
        self.motion_configs = self.add_argument_group('Motion Transformer')
        self.motion_configs.add_argument('--motion_hidden_dim', default=256, type=int, help='Hidden dimension after initial motion embedding')
        self.motion_configs.add_argument('--motion_latent_dim', default=256, type=int, help='Latent dimension in motion encoder')
        self.motion_configs.add_argument('--motion_n_heads', default=8, type=int, help='Number of attention heads')
        self.motion_configs.add_argument('--motion_n_layers', default=3, type=int, help='Number of transformer layers')
        self.motion_configs.add_argument('--motion_intermediate_dim', default=1024, type=int, help='Intermediate dimension in FFN')
        self.motion_configs.add_argument('--dropout', default=0.0, type=float, help='Dropout rate')

        # === Cross-Modal Configuration (Based on GIMO) ===
        self.cross_modal_configs = self.add_argument_group('Cross-Modal Transformer')
        self.cross_modal_configs.add_argument('--cross_n_heads', default=8, type=int)
        self.cross_modal_configs.add_argument('--cross_hidden_dim', default=256, type=int)
        self.cross_modal_configs.add_argument('--cross_intermediate_dim', default=1024, type=int)
        self.cross_modal_configs.add_argument('--cross_n_layers', default=3, type=int)

        # === Training Configuration ===
        self.train_configs = self.add_argument_group('Training')
        self.train_configs.add_argument('--save_path', type=str, default='chkpoints_adt/', help='Directory to save checkpoints')
        self.train_configs.add_argument('--save_fre', type=int, default=10, help='Checkpoint saving frequency (epochs)')
        # self.train_configs.add_argument('--vis_fre', type=int, default=1000) # Maybe redefine for trajectory viz
        self.train_configs.add_argument('--val_fre', type=int, default=10, help='Validation frequency (epochs)')
        self.train_configs.add_argument('--num_val_visualizations', type=int, default=10, help='Number of samples to visualize during validation')
        self.train_configs.add_argument('--load_model_dir', type=str, default=None, help='Path to load a pretrained model checkpoint')
        self.train_configs.add_argument('--load_optim_dir', type=str, default=None, help='Path to load optimizer state separately')

        self.train_configs.add_argument('--epoch', type=int, default=200, help='Total training epochs')
        self.train_configs.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.train_configs.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
        self.train_configs.add_argument('--gamma', type=float, default=0.99, help='Learning rate decay factor')
        # Redefine loss lambdas if needed for motion prediction
        # self.train_configs.add_argument('--lambda_...', type=float, default=1)
        self.train_configs.add_argument('--lambda_path_xyz', type=float, default=1.0, help='Weight for the L1 path loss on XYZ coordinates')
        self.train_configs.add_argument('--lambda_dest_xyz', type=float, default=1.0, help='Weight for the L1 destination loss on XYZ coordinates')
        self.train_configs.add_argument('--lambda_rec_xyz', type=float, default=1.0, help='Weight for the L1 reconstruction loss on history XYZ coordinates')

        # === Wandb Configuration ===
        self.wandb_configs = self.add_argument_group('WandB')
        self.wandb_configs.add_argument('--wandb_project', type=str, default="GIMO_ADT_Motion", help='Wandb project name')
        self.wandb_configs.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity (username)')
        self.wandb_configs.add_argument('--wandb_mode', type=str, default="online", choices=["online", "offline", "disabled"], help='Wandb mode')

        # === Evaluation Configuration ===
        self.eval_configs = self.add_argument_group('Evaluation')
        self.eval_configs.add_argument('--output_path', default='results_adt/', type=str, help='Directory to save evaluation outputs')
        self.eval_configs.add_argument('--comment', default='', type=str, help='Custom comment for evaluation run')

    def get_configs(self):
        # Basic validation
        args = self.parse_args()
        return args


if __name__ == '__main__':
    adt_config = ADTObjectMotionConfig()
    print(adt_config.get_configs()) 