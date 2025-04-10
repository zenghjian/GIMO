python train.py --dataroot /usr/data/cvpr_shared/saroha/GIMO --save_path ./chkpoints \
 --val_fre 1 --seq_decay_ratio 1 --gaze_points 10 --batch_size 4 --sample_points 300000 \
 --motion_n_layers 6 --gaze_n_layers 6 --gaze_latent_dim 256 --cross_hidden_dim 256 \
 --cross_n_layers 6 --num_workers 4 --output_path ./results/gaze3d_train_scratch \
#  --disable_gaze --disable_crossmodal


# python train.py --dataroot ./demo_training_data --save_path ./chkpoints \
#  --val_fre 1 --seq_decay_ratio 1 --gaze_points 10 --batch_size 2 --sample_points 10000 \
#  --motion_n_layers 6 --gaze_n_layers 6 --gaze_latent_dim 256 --cross_hidden_dim 256 \
#  --cross_n_layers 6 --num_workers 8 \
#  --disable_gaze --disable_crossmodal


# python train.py --dataroot ./demo_training_data --save_path ./chkpoints \
#  --val_fre 1 --seq_decay_ratio 1 --gaze_points 10 --batch_size 2 --sample_points 10000 \
#  --motion_n_layers 6 --gaze_n_layers 6 --gaze_latent_dim 256 --cross_hidden_dim 256 \
#  --cross_n_layers 6 --num_workers 8 \
# #  --disable_gaze --disable_crossmodal
