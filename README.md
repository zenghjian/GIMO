```bash
python train_adt_overfitting.py \
    --adt_dataroot <path_to_adt_data> \
    --save_path results/<experiment_name> \
    --comment <experiment_name> \
    --epoch 10000 \
    --save_fre 1000 \
    --val_fre 1000 \
    --train_ratio 1.0 \
    --weight_decay 0.0 \
    --gamma 1.0 

Key Parameters (`config/adt_config.py`):

*   `--use_first_frame_only`: Default: `False`. If `True`, use only the first frame's pose as input to predict the entire subsequent trajectory.

*   `--history_fraction`: Default: `0.3`. Fraction of the total trajectory length (`trajectory_length`) used as history input. Only effective when `--use_first_frame_only` is `False`.

*   `--pointcloud_subsample`: Default: `1`. Factor to downsample the full scene point cloud during loading (e.g., `10` uses 1/10th of the points). `1` means no downsampling.

*   `--trajectory_pointcloud_radius`: Default: `1.0`. Radius (in meters) around the trajectory used to filter the scene point cloud for each trajectory sample (creating trajectory-specific point clouds).

*   `--sample_points`: Default: `50000`. Target number of points per point cloud sample in a batch. If a source point cloud has fewer points, points are sampled *with replacement* to reach this number. If more, points are randomly sampled *without replacement*.

*   `--no_text_embedding`: Default: `False`. If `True`, disable the use of object category text embeddings in the model.

*   `--lambda_trans`: Default: `1.0`. Weight for the translation (position) component of the loss.

*   `--lambda_ori`: Default: `1.0`. Weight for the orientation component of the loss.

*   `--lambda_rec`: Default: `1.0`. Weight for the reconstruction loss (applied to the history part when not using `--use_first_frame_only`).

Example command for overfitting:

