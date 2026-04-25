# Configuration Files

This directory contains YAML configuration files for training 3D medical image segmentation models.

## Available Configurations

- `host.yaml` - Configuration for local development (CPU)
- `remote.yaml` - Configuration for remote training server (CUDA)
- `optimized_remote.yaml` - Optimized configuration with attention mechanisms

## Training Script Usage

### Basic Training

```bash
# Train with a specific config file
python script/train_split.py --config config/host.yaml

# Train risk prediction model (cross-attention, tabular)
python script/train_risk.py --config config/host.yaml

# Train on remote server
python script/train_split.py --config config/remote.yaml

# Train with optimized model
python script/train_split.py --config config/optimized_remote.yaml
```

### Advanced Options

```bash
# Resume training from checkpoint
python script/train_split.py --config config/remote.yaml --resume checkpoints/latest.pth

# Override device
python script/train_split.py --config config/host.yaml --device cuda

# Combine options
python script/train_split.py \
    --config config/remote.yaml \
    --resume checkpoints/best.pth \
    --device cuda
```

### Command Line Arguments

- `--config` (required): Path to YAML configuration file
- `--resume` (optional): Path to checkpoint to resume training from
- `--device` (optional): Override device from config (cuda/cpu)

## Configuration Structure

Each YAML config file contains:

- **model**: Model architecture parameters
  - `name`: Model identifier
  - `in_channels`: Number of input channels
  - `out_channels`: Number of output channels
  - `base_filters`: Base number of filters
  - `depth`: Network depth
  - `use_residual`: Use residual connections
  - `use_attention`: Use attention gates
  - `dropout`: Dropout rate
  - `norm_type`: Normalization type (batch/instance/group)
  - `activation`: Activation function

- **train**: Training parameters
  - `epochs`: Number of training epochs
  - `batch_size`: Batch size
  - `learning_rate`: Learning rate
  - `weight_decay`: Weight decay for regularization
  - `patch_size`: Size of 3D patches [D, H, W]
  - `max_patches_per_volume`: Maximum patches per volume
  - `optimizer`: Optimizer type (adam/adamw/sgd)
  - `scheduler`: Learning rate scheduler
  - `warmup_epochs`: Warmup period

- **data**: Dataset configuration
  - `train_dirs`: List of training data directories
  - `eval_dirs`: List of evaluation data directories
  - `num_workers`: Number of data loading workers
  - `prefetch_factor`: Prefetch factor for DataLoader
  - `file_patterns`: Patterns to match image files
  - `label_suffix`: Suffix for label files
  - `train_ratio`: Ratio of training data

- **async_load**: Memory management
  - `ram_cache_gb`: RAM cache size in GB
  - `vram_batch_gb`: VRAM batch size in GB
  - `auto_config`: Enable automatic configuration

- **compile**: PyTorch compilation
  - `enabled`: Enable torch.compile
  - `mode`: Compilation mode
  - `fullgraph`: Full graph compilation

- **checkpoint**: Checkpoint settings
  - `save_dir`: Directory to save checkpoints
  - `save_best`: Save best model
  - `save_interval`: Save interval (epochs)
  - `max_keep`: Maximum checkpoints to keep

- **device**: Computing device (cuda/cpu)
- **seed**: Random seed for reproducibility
- **log_dir**: Logging directory
- **eval_interval`: Evaluation interval (epochs)

- **risk**: Tabular risk prediction configuration
  - `enabled`: Enable risk task entrypoint
  - `excel_path`: XLSX source file path
  - `use_cleaned_data`: Enable cleaning before training
  - `use_dummy_metadata`: Enable metadata-only dummy training mode
  - `auto_max_features`: Maximum selected numeric features
  - `feature_select_method`: Feature selection algorithm

## Platform-Specific Configurations

### REMOTE CONFIG hyperai platform

40GB RAM, 5090

train:
    - "/hyperai/input/input0/singlecrop1"
    - "/hyperai/input/input0/singlecrop2"
    - "/hyperai/input/input0/singlecrop3"
    - "/hyperai/input/input0/singlecrop4"
    - "/hyperai/input/input0/MultiCrop"

eval:
    - "/hyperai/input/input0/Multi"