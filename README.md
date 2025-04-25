# GIMO for ADT Object Motion Prediction

This project adapts the GIMO (Generalizable Implicit Motion) architecture to predict the 3D motion of objects within Aria Digital Twin (ADT) sequences. It takes ADT scene data, including object trajectories and point clouds, as input and outputs predictions for future object motion.

## Features

*   **3D Object Motion Prediction:** Predicts future 3D coordinates (XYZ) of objects based on their historical motion and scene context.
*   **GIMO-Inspired Architecture:** Utilizes a transformer-based model incorporating:
    *   A scene encoder (PointNet-based) to process static scene point clouds.
    *   A motion encoder/decoder transformer to model object dynamics.
    *   Cross-modal attention mechanisms to fuse scene and motion information.
*   **ADT Data Compatibility:** Designed to work directly with ADT sequence data format.
*   **Flexible Data Loading:**
    *   Supports automatic scanning of directories containing multiple ADT sequences.
    *   Allows specifying pre-defined training and validation splits using text files (`--train_split_file`, `--val_split_file`).
    *   Falls back to dynamic train/validation splitting based on a ratio (`--train_ratio`) if split files are not provided (requires `adt_sequence_utils`).
*   **Point Cloud Processing:** Extracts and utilizes scene point clouds for contextual understanding.
*   **Trajectory Caching:** Caches processed trajectory data for faster loading during subsequent runs (`--use_cache`).
*   **Motion Filtering:**
    *   Filters static or minimally moving trajectories based on a threshold (`--min_motion_threshold`).
    *   Optionally filters based on motion percentile (`--min_motion_percentile`).
*   **Motion Segment Detection (Optional):** Can detect and extract active motion segments within longer trajectories (`--detect_motion_segments`).
*   **Validation & Visualization:** Includes validation loops and generates visualizations of ground truth and predicted trajectories.
*   **Weights & Biases Integration:** Logs metrics, configurations, and visualizations to WandB for experiment tracking (`--wandb_project`, `--wandb_entity`, `--wandb_mode`).
*   **Configuration:** Highly configurable through command-line arguments (see `config/adt_config.py`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\\Scripts\\activate  # Windows
    ```
    Or using Conda:
    ```bash
    conda create -n gimo_adt python=3.9
    conda activate gimo_adt
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio numpy tqdm wandb
    ```
    *(Consider creating a `requirements.txt` file)*

## Training

Use the `train_adt.py` script to train the motion prediction model.

**Basic Command Structure:**

```bash
python train_adt.py --adt_dataroot <path_to_data> --save_path <path_to_save_checkpoints> [OPTIONS]
```

**Key Arguments:**

*   `--adt_dataroot`: Path to the root directory containing ADT sequences OR path to a single sequence file/directory (used for scanning if split files aren't provided).
*   `--save_path`: Directory where checkpoints, logs, and visualizations will be saved.
*   `--train_split_file`: (Optional) Path to the text file listing training sequence paths.
*   `--val_split_file`: (Optional) Path to the text file listing validation sequence paths.
*   `--batch_size`: Training batch size.
*   `--epoch`: Number of training epochs.
*   `--lr`: Learning rate.
*   `--trajectory_length`: Length of trajectory segments to extract.
*   `--history_fraction`: Fraction of trajectory length used as input history (e.g., 0.3 means 30% history, 70% future prediction).
*   `--load_pointcloud`: Enable/disable loading scene point clouds (default: True).
*   `--sample_points`: Number of points to sample from the scene point cloud.
*   `--wandb_project`: Name of the WandB project.
*   `--wandb_entity`: Your WandB username or team name.
*   `--wandb_mode`: `online`, `offline`, or `disabled`.

**Example (Dynamic Splitting - requires `adt_sequence_utils`):**

```bash
python train_adt.py \
    --adt_dataroot ./data/adt_sequences \
    --save_path ./chkpoints_adt \
    --epoch 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --train_ratio 0.9 \
    --trajectory_length 100 \
    --history_fraction 0.3 \
    --sample_points 3000 \
    --wandb_project "ADT_Motion_Prediction" \
    --wandb_mode online
```

**Example (Using Split Files):**

```bash
python train_adt.py \
    --adt_dataroot ./data/adt_sequences \
    --train_split_file ./data/splits/train.txt \
    --val_split_file ./data/splits/val.txt \
    --save_path ./chkpoints_adt_split \
    --epoch 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --trajectory_length 100 \
    --history_fraction 0.3 \
    --sample_points 3000 \
    --wandb_project "ADT_Motion_Prediction" \
    --wandb_mode online
```

## Evaluation

Evaluation is performed using the `evaluate_gimo_adt.py` script (details depend on its implementation). It typically involves:

1.  Loading a trained model checkpoint (`--load_model_dir`).
2.  Specifying the dataset (`--adt_dataroot` and potentially `--val_split_file` or a dedicated test split file).
3.  Running inference and calculating evaluation metrics (e.g., ADE, FDE).
4.  Saving results to an output directory (`--output_path`).

**Example (Conceptual):**

```bash
python evaluate_gimo_adt.py \
    --load_model_dir ./chkpoints_adt/best_model.pth \
    --adt_dataroot ./data/adt_sequences \
    --output_path ./results_adt \
    --batch_size 8
```

## Configuration Options

For a detailed list of all configuration options and their descriptions, please refer to the `ADTObjectMotionConfig` class in `config/adt_config.py`. Key argument groups include:

*   **Input:** Batch size, data loader workers.
*   **Trajectory & Dataset:** Data paths, split configurations, trajectory parameters (length, history, skipping), point cloud settings, caching, motion filtering/segmentation.
*   **Scene Encoder:** PointNet configuration (feature dimensions, sampling).
*   **Motion Transformer:** Transformer architecture details (dimensions, heads, layers).
*   **Cross-Modal Transformer:** Configuration for the cross-attention module.
*   **Training:** Checkpoint saving, validation frequency, visualization count, resuming options, optimizer/scheduler settings, loss weights.
*   **WandB:** Project/entity/mode settings.
*   **Evaluation:** Output path for evaluation results.

## Output Files

During training, the following files and directories are created under the specified `--save_path`:

*   `chkpt_epoch_*.pth`: Checkpoints saved periodically based on `--save_fre`.
*   `best_model.pth`: Model checkpoint with the lowest validation loss achieved so far.
*   `final_model.pth`: Model checkpoint saved after the last training epoch.
*   `train_log.txt`: Text file containing training and validation logs.
*   `train_sequences.txt`: List of sequence paths used for training (saved regardless of whether split files were provided).
*   `val_sequences.txt`: List of sequence paths used for validation (saved regardless of whether split files were provided).
*   `val_visualizations/`: Directory containing trajectory visualizations generated during validation loops (organized by epoch).
    *   `epoch_*/{obj_name}_trajectory_split.png`: Ground truth history/future split.
    *   `epoch_*/{obj_name}_prediction_vs_gt_epoch{epoch}.png`: Prediction vs. Ground Truth.
    *   `epoch_*/{obj_name}_full_trajectory.png`: Full Ground Truth trajectory.
*   `trajectory_cache/`: Stores cached trajectory data (`.pkl` files) if `--use_cache` is enabled.

The evaluation script (`evaluate_gimo_adt.py`) will save its outputs (metrics, potentially predicted trajectories) to the directory specified by its `--output_path` argument.

## Dependencies

*   Python (>= 3.8 recommended)
*   PyTorch (>= 1.10)
*   NumPy
*   tqdm
*   WandB
*   (Optional) `adt_sequence_utils` from `ariaworldgaussians`

*(Create a `requirements.txt` file for easier installation)* 