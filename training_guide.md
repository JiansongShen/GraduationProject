# Training Script Quick Start Guide

> - by GPT5.5

## Overview
The `train_split.py` script provides a flexible way to train 3D medical image segmentation models using configuration files.

## Basic Usage

### 1. Train with a config file
```bash
python script/train_split.py --config config/host.yaml
```

### 2. Train on remote server
```bash
python script/train_split.py --config config/remote.yaml
```

### 3. Resume from checkpoint
```bash
python script/train_split.py --config config/remote.yaml --resume checkpoints/latest.pth
```

### 4. Override device
```bash
python script/train_split.py --config config/host.yaml --device cuda
```

## Command Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--config` | Yes | Path to YAML config file | `config/remote.yaml` |
| `--resume` | No | Checkpoint path to resume from | `checkpoints/best.pth` |
| `--device` | No | Override device (cuda/cpu) | `cuda` |

## What the Script Does

1. **Loads configuration** from YAML file
2. **Builds model** (AttentionUnet) based on config
3. **Sets up data loaders** with patch-based sampling
4. **Trains the model** with combined BCE + Dice loss
5. **Evaluates periodically** on validation set
6. **Saves checkpoints** (latest and best)
7. **Logs metrics** to TensorBoard

## Output Files

After training, you'll find:

```
checkpoints/
├── latest.pth      # Most recent checkpoint
└── best.pth        # Best performing model

logs/
├── training.log    # Training log file
└── tensorboard/    # TensorBoard logs
    └── events.out.tfevents...
```

## Monitoring Training

### View TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006 in your browser

### Check Training Log
```bash
tail -f logs/training.log
```

## Key Features

✅ **Configurable**: All parameters via YAML files  
✅ **Checkpointing**: Auto-save best and latest models  
✅ **Resume Training**: Continue from any checkpoint  
✅ **TensorBoard Logging**: Visualize training progress  
✅ **Multi-Optimizer Support**: Adam, AdamW, SGD  
✅ **LR Scheduling**: Cosine annealing, step LR  
✅ **Combined Loss**: BCE + Dice for better segmentation  
✅ **Device Auto-Detection**: Falls back to CPU if no GPU  
✅ **Reproducible**: Random seed control  

## Tips

1. **Start small**: Use `host.yaml` for testing on CPU
2. **Monitor VRAM**: Adjust `batch_size` and `patch_size` if OOM
3. **Use checkpoints**: Resume from `latest.pth` if interrupted
4. **Check TensorBoard**: Monitor loss and dice score trends
5. **Adjust learning rate**: If loss doesn't decrease, try lower LR

## Common Issues

### Out of Memory
- Reduce `batch_size` in config
- Reduce `patch_size` 
- Increase `num_workers` for better memory management

### Slow Training
- Enable `compile.enabled: true` if using PyTorch 2.0+
- Increase `batch_size` if you have enough VRAM
- Use `optimized_remote.yaml` settings

### Loss Not Decreasing
- Check if data paths are correct
- Verify labels exist and are properly formatted
- Try lower learning rate
- Increase warmup epochs

## Next Steps

After training completes:
1. Check `checkpoints/best.pth` for the best model
2. Review TensorBoard logs for training curves
3. Use the trained model for inference
4. Fine-tune hyperparameters if needed

## Example: Full Training Run

```bash
# 1. Test on CPU first
python script/train_split.py --config config/host.yaml

# 2. Train on GPU
python script/train_split.py --config config/remote.yaml

# 3. Monitor in another terminal
tensorboard --logdir logs/tensorboard

# 4. If interrupted, resume
python script/train_split.py --config config/remote.yaml --resume checkpoints/latest.pth
```

Happy Training! 🚀
