from argparse import ArgumentParser, ArgumentTypeError

class ADTObjectMotionConfig(ArgumentParser):
    def __init__(self):
        super().__init__()

        # === Input Data Configuration ===
        self.input_configs = self.add_argument_group('Input')
        self.input_configs.add_argument('--batch_size', default=1, type=int, help='Batch size for training/evaluation')
        self.input_configs.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading')

        # === Trajectory & Dataset Configuration ===
        self.traj_dataset_configs = self.add_argument_group('Trajectory & Dataset')
        self.traj_dataset_configs.add_argument('--adt_dataroot', default='./data', type=str, help='Path to ADT sequence or directory containing sequences/split files')
        self.traj_dataset_configs.add_argument('--train_split_file', default=None, type=str, help='Path to train split file (overrides dataroot scan/split)')
        self.traj_dataset_configs.add_argument('--val_split_file', default=None, type=str, help='Path to validation split file (overrides dataroot scan/split)')
        self.traj_dataset_configs.add_argument('--train_ratio', default=0.9, type=float, help='Ratio of sequences to use for training (default: 0.9, used if split files not provided)')
        self.traj_dataset_configs.add_argument('--split_seed', default=42, type=int, help='Random seed for train/val split (used if split files not provided)')
        self.traj_dataset_configs.add_argument('--trajectory_length', default=200, type=int, help='Total length of trajectory segments extracted (before split)')
        self.traj_dataset_configs.add_argument('--history_fraction', default=0.3, type=float, help='Fraction of trajectory_length to use for history (e.g., 0.6 for 3/5 ratio)')
        self.traj_dataset_configs.add_argument('--object_position_dim', default=3, type=int, help='Dimension of object position (x, y, z)')
        self.traj_dataset_configs.add_argument('--object_rotation_dim', default=6, type=int, help='Dimension of object rotation (6D representation)')
        self.traj_dataset_configs.add_argument('--object_motion_dim', default=9, type=int, help='Total dimension of object motion (positions + rotations)')
        self.traj_dataset_configs.add_argument('--point_cloud_dim', default=3, type=int, help='Dimension of input point cloud points (e.g., 3 for XYZ)')
        self.traj_dataset_configs.add_argument('--skip_frames', default=5, type=int, help='Frames to skip when extracting trajectory points from dataset')
        self.traj_dataset_configs.add_argument('--use_displacements', default=False, action='store_true', help='Use relative displacements instead of absolute positions')
        self.traj_dataset_configs.add_argument('--use_first_frame_only', default=False, action='store_true', help='Use only the first frame as input, predict full sequence')
        self.traj_dataset_configs.add_argument('--load_pointcloud', default=True, action='store_true', help='Enable loading point cloud data from sequences')
        self.traj_dataset_configs.add_argument('--pointcloud_subsample', default=1, type=int, help='Subsample factor for point cloud (higher means fewer points)')
        self.traj_dataset_configs.add_argument('--trajectory_pointcloud_radius', default=1.0, type=float, help='Radius (in meters) around trajectory to collect points for trajectory-specific point clouds')
        self.traj_dataset_configs.add_argument('--min_motion_threshold', default=0.5, type=float, help='Minimum motion threshold in meters for trajectory filtering')
        # 0.5 for diff
        self.traj_dataset_configs.add_argument('--min_motion_percentile', default=0.0, type=float, help='Filter trajectories below this percentile of motion')
        self.traj_dataset_configs.add_argument('--no_use_cache', dest='use_cache', action='store_false', help='Disable caching for trajectories')
        self.traj_dataset_configs.add_argument('--detect_motion_segments', default=False, action='store_true', help='Enable detection and extraction of active motion segments')
        self.traj_dataset_configs.add_argument('--motion_velocity_threshold', default=0.05, type=float, help='Threshold in m/s for detecting active motion')
        self.traj_dataset_configs.add_argument('--min_segment_frames', default=5, type=int, help='Minimum number of frames for a valid motion segment')
        self.traj_dataset_configs.add_argument('--max_stationary_frames', default=3, type=int, help='Maximum consecutive stationary frames allowed in a motion segment')
        self.traj_dataset_configs.add_argument('--normalize_data', default=False, action='store_true', help='Enable normalization of trajectory data using scene bounds')
        self.traj_dataset_configs.add_argument('--global_cache_dir', type=str, default=None, help='Path to a shared global directory for trajectory cache (overrides cache within save_path)')
        self.traj_dataset_configs.add_argument('--force_use_cache', action='store_true', default=False, help='Force using available cache files even if parameters don\'t match')

        # === Scene/Point Cloud Configuration ===
        self.scene_configs = self.add_argument_group('Scene Encoder')
        self.scene_configs.add_argument('--scene_feats_dim', default=256, type=int, help='Output dimension of the PointNet scene encoder')
        self.scene_configs.add_argument('--sample_points', default=50000, type=int, help='Number of points to sample from the scene point cloud') 
        self.scene_configs.add_argument('--no_bbox', action='store_true', default=False, help='Disable bounding box processing and conditioning')
        self.scene_configs.add_argument('--no_scene', action='store_true', default=False, help='Disable scene global features and use only motion features')
        self.scene_configs.add_argument('--no_semantic_bbox', action='store_true', default=False, help='Disable semantic bbox embedding for scene conditioning')
        self.scene_configs.add_argument('--no_semantic_text', action='store_true', default=False, help='Disable semantic text embedding for scene conditioning')
        self.scene_configs.add_argument('--max_bboxes', default=400, type=int, help='Maximum number of bounding boxes for scene conditioning')
        self.scene_configs.add_argument('--semantic_bbox_embed_dim', default=256, type=int, help='Output dimension of semantic bbox embedder')
        self.scene_configs.add_argument('--semantic_text_embed_dim', default=256, type=int, help='Output dimension of semantic text embedder')
        self.scene_configs.add_argument('--semantic_bbox_hidden_dim', default=128, type=int, help='Hidden dimension in semantic bbox embedder')
        self.scene_configs.add_argument('--semantic_bbox_num_heads', default=4, type=int, help='Number of attention heads in semantic bbox embedder')
        self.scene_configs.add_argument('--semantic_bbox_use_attention', action='store_true', default=True, help='Use attention in semantic bbox embedder')
        # self.scene_configs.add_argument('--pointnet_chkpoints', default='pretrained/point.model', type=str, help='Path to pretrained PointNet weights') # Consider adding back later

        # === Motion Pathway Configuration (Based on GIMO) ===
        self.motion_configs = self.add_argument_group('Motion Transformer')
        self.motion_configs.add_argument('--motion_hidden_dim', default=256, type=int, help='Hidden dimension after initial motion embedding')
        self.motion_configs.add_argument('--motion_latent_dim', default=256, type=int, help='Latent dimension in motion encoder')
        self.motion_configs.add_argument('--motion_n_heads', default=8, type=int, help='Number of attention heads')
        self.motion_configs.add_argument('--motion_n_layers', default=3, type=int, help='Number of transformer layers')
        self.motion_configs.add_argument('--motion_intermediate_dim', default=1024, type=int, help='Intermediate dimension in FFN')
        self.motion_configs.add_argument('--dropout', default=0.0, type=float, help='Dropout rate')

        # === Output Pathway Configuration ===
        self.output_configs = self.add_argument_group('Output Pathway')
        self.output_configs.add_argument('--embedding_hidden_dim', default=256, type=int, help='Hidden dimension for the embedding layer')
        self.output_configs.add_argument('--output_latent_dim', default=256, type=int, help='Latent dimension in output encoder')
        self.output_configs.add_argument('--output_n_heads', default=8, type=int, help='Number of attention heads in output encoder')
        self.output_configs.add_argument('--output_n_layers', default=3, type=int, help='Number of transformer layers in output encoder')
        
        # === Text/Category Embedding Configuration ===
        self.text_configs = self.add_argument_group('Text/Category Embedding')
        self.text_configs.add_argument('--no_text_embedding', action='store_true', default=False, help='Disable text/category embedding')
        self.text_configs.add_argument('--category_embed_dim', default=256, type=int, help='Dimension of category embedding')
        self.text_configs.add_argument('--num_object_categories', default=50, type=int, help='Number of object categories (for legacy torch.embedding)')
        self.text_configs.add_argument('--clip_model_name', default="ViT-B/32", type=str, choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"], help='CLIP model to use for category embedding')
        self.text_configs.add_argument('--use_legacy_category_embedding', action='store_true', default=False, help='Use legacy torch.embedding instead of CLIP for category embedding')

        # === End Pose Configuration ===
        self.end_pose_configs = self.add_argument_group('End Pose Conditioning')
        self.end_pose_configs.add_argument('--no_end_pose', action='store_true', default=False, help='Disable end pose embedding and conditioning')
        self.end_pose_configs.add_argument('--end_pose_embed_dim', default=256, type=int, help='Dimension of end pose embedding')

        # === Training Configuration ===
        self.train_configs = self.add_argument_group('Training')
        self.train_configs.add_argument('--save_path', type=str, default='chkpoints_adt/', help='Directory to save checkpoints')
        self.train_configs.add_argument('--save_fre', type=int, default=10, help='Checkpoint saving frequency (epochs)')
        # self.train_configs.add_argument('--vis_fre', type=int, default=1000) # Maybe redefine for trajectory viz
        self.train_configs.add_argument('--val_fre', type=int, default=1, help='Validation frequency (epochs)')
        self.train_configs.add_argument('--num_val_visualizations', type=int, default=1, help='Number of samples to visualize during validation, 0 means no visualization')
        self.train_configs.add_argument('--load_model_dir', type=str, default=None, help='Path to load a pretrained model checkpoint')
        self.train_configs.add_argument('--load_optim_dir', type=str, default=None, help='Path to load optimizer state separately')
        self.train_configs.add_argument('--timestep', type=int, default=0, help='Timestep for autoregressive prediction ')

        self.train_configs.add_argument('--epoch', type=int, default=200, help='Total training epochs')
        self.train_configs.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.train_configs.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
        self.train_configs.add_argument('--gamma', type=float, default=0.99, help='Learning rate decay factor')
        self.train_configs.add_argument('--adam_beta1', type=float, default=0.9, help='AdamW optimizer beta1 parameter')
        self.train_configs.add_argument('--adam_beta2', type=float, default=0.999, help='AdamW optimizer beta2 parameter')
        self.train_configs.add_argument('--adam_eps', type=float, default=1e-8, help='AdamW optimizer epsilon parameter')
        
        # Loss weights for position and orientation components
        self.train_configs.add_argument('--lambda_trans', type=float, default=1.0, help='Weight for translation/position loss')
        self.train_configs.add_argument('--lambda_ori', type=float, default=1.0, help='Weight for orientation loss')
        self.train_configs.add_argument('--lambda_rec', type=float, default=1.0, help='Weight for reconstruction loss')
        
        # Gradient tracking and debugging parameters
        self.train_configs.add_argument('--enable_gradient_tracking', action='store_true', default=False, help='Enable gradient tracking during training')
        self.train_configs.add_argument('--gradient_log_freq', type=int, default=10, help='Frequency of gradient logging (batches)')
        self.train_configs.add_argument('--gradient_plot_freq', type=int, default=None, help='Frequency of gradient plotting (epochs, defaults to val_fre)')
        self.train_configs.add_argument('--enable_orientation_analysis', action='store_true', default=False, help='Enable orientation distribution analysis before training')
        self.train_configs.add_argument('--orientation_analysis_samples', type=int, default=500, help='Number of samples for orientation analysis')

        # === Wandb Configuration ===
        self.wandb_configs = self.add_argument_group('WandB')
        self.wandb_configs.add_argument('--wandb_project', type=str, default="GIMO_ADT_Motion", help='Wandb project name')
        self.wandb_configs.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity (username)')
        self.wandb_configs.add_argument('--wandb_mode', type=str, default="online", choices=["online", "offline", "disabled"], help='Wandb mode')

        # === Evaluation Configuration ===
        self.eval_configs = self.add_argument_group('Evaluation')
        self.eval_configs.add_argument('--output_path', default='results_adt/', type=str, help='Directory to save evaluation outputs')
        
        # === Visualization Configuration ===
        self.viz_configs = self.add_argument_group('Visualization')
        self.viz_configs.add_argument('--viz_ori_scale', type=float, default=0.2, help='Scale factor for orientation arrows in visualization')
        self.viz_configs.add_argument('--show_ori_arrows', action='store_true', default=False, help='Show orientation arrows in visualization')
        self.viz_configs.add_argument('--visualize_train_trajectories_on_start', action='store_true', default=False, help='Generate and save visualizations for a subset of the training data at the beginning of training.')

        # === DiT Configuration ===
        self.dit_configs = self.add_argument_group('Diffusion Transformer (DiT)')
        self.dit_configs.add_argument('--use_dit', action='store_true', default=False, help='Use GIMO Multimodal DiT instead of standard GIMO model')
        self.dit_configs.add_argument('--dit_patch_size', type=int, default=8, help='Patch size for DiT (trajectory will be divided into patches)')
        self.dit_configs.add_argument('--dit_hidden_dim', type=int, default=384, help='Hidden dimension for DiT transformer blocks')
        self.dit_configs.add_argument('--dit_depth', type=int, default=12, help='Number of DiT transformer blocks')
        self.dit_configs.add_argument('--dit_num_heads', type=int, default=6, help='Number of attention heads in DiT blocks')
        self.dit_configs.add_argument('--dit_mlp_ratio', type=float, default=4.0, help='MLP expansion ratio in DiT blocks')
        self.dit_configs.add_argument('--dit_time_embed_dim', type=int, default=640, help='Timestep embedding dimension for DiT')
        self.dit_configs.add_argument('--dit_cond_embed_dim', type=int, default=640, help='Conditioning embedding dimension for DiT')
        self.dit_configs.add_argument('--dit_cond_dim', type=int, default=512, help='Output dimension of multimodal conditioning features')
        self.dit_configs.add_argument('--dit_cond_heads', type=int, default=8, help='Number of attention heads in conditioning encoder')
        self.dit_configs.add_argument('--dit_cond_layers', type=int, default=3, help='Number of layers in conditioning encoder')
        self.dit_configs.add_argument('--dit_perceiver_latent_size', type=int, default=64, help='Size of perceiver latent array for multimodal conditioning')
        
        # === Diffusion Training Configuration ===
        self.diffusion_configs = self.add_argument_group('Diffusion Training')
        self.diffusion_configs.add_argument('--diffusion_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
        self.diffusion_configs.add_argument('--diffusion_beta_start', type=float, default=1e-4, help='Starting value for noise schedule')
        self.diffusion_configs.add_argument('--diffusion_beta_end', type=float, default=0.02, help='Ending value for noise schedule')
        self.diffusion_configs.add_argument('--diffusion_loss_type', type=str, default='l2', choices=['l1', 'l2'], help='Loss function for diffusion training')
        self.diffusion_configs.add_argument('--diffusion_sampling_steps', type=int, default=100, help='Number of sampling steps during inference (can be less than training timesteps)')

        # === BPS Encoder Configuration ===
        self.bps_configs = self.add_argument_group('BPS Encoder')
        self.bps_configs.add_argument('--use_bps', action='store_true', default=False, help='Enable BPS encoder for object shape conditioning')
        self.bps_configs.add_argument('--bps_input_dim', default=1024*3, type=int, help='Input dimension for BPS encoder (1024 points * 3 coordinates)')
        self.bps_configs.add_argument('--bps_hidden_dim', default=512, type=int, help='Hidden dimension in BPS encoder')
        self.bps_configs.add_argument('--bps_output_dim', default=256, type=int, help='Output dimension of BPS encoder')
        self.bps_configs.add_argument('--bps_num_points', default=1024, type=int, help='Number of BPS points to sample from bbox')
        self.bps_configs.add_argument('--lambda_obj_geo', type=float, default=1.0, help='Weight for object geometry loss')

        # === CHOIS Sparse Conditioning Configuration ===
        self.sparse_cond_configs = self.add_argument_group('Sparse Conditioning')
        self.sparse_cond_configs.add_argument('--conditioning_strategy', type=str, default='history_fraction', choices=['full_trajectory', 'history_fraction', 'chois_original'], help='Strategy for creating trajectory conditions')
        self.sparse_cond_configs.add_argument('--waypoint_interval', default=10, type=int, help='Frame interval for waypoints in sparse conditioning (for chois_original strategy)')
    def get_configs(self):
        # Basic validation
        args = self.parse_args()
        
        # Ensure object_motion_dim matches position_dim + orientation_dim
        if args.object_motion_dim != args.object_position_dim + args.object_rotation_dim:
            print(f"Warning: object_motion_dim ({args.object_motion_dim}) does not match "
                  f"object_position_dim ({args.object_position_dim}) + "
                  f"object_rotation_dim ({args.object_rotation_dim}). "
                  f"Setting object_motion_dim = position_dim + rotation_dim.")
            args.object_motion_dim = args.object_position_dim + args.object_rotation_dim
        
        # Validate DiT patch size
        if args.use_dit:
            if args.trajectory_length % args.dit_patch_size != 0:
                print(f"Warning: trajectory_length ({args.trajectory_length}) is not divisible by "
                      f"dit_patch_size ({args.dit_patch_size}). This may cause issues with DiT.")
                # Suggest a compatible patch size
                for patch_size in [8, 4, 2, 1]:
                    if args.trajectory_length % patch_size == 0:
                        print(f"Suggested dit_patch_size: {patch_size}")
                        break
            
        return args


if __name__ == '__main__':
    adt_config = ADTObjectMotionConfig()
    print(adt_config.get_configs()) 