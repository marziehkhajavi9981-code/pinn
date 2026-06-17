# PINN Velocity Reconstruction

This repository trains physics-informed neural networks for 2D velocity-field
reconstruction from `u_stack.npz` and `v_stack.npz`.

## Command-Line Guide

### Generate Masks And Splits

Create a sparse observation mask plus train/val/test split artifact:

```bash
python scripts/make_masks_splits.py \
  --u u_stack.npz \
  --v v_stack.npz \
  --crop_h 64 \
  --crop_w 64 \
  --mask_model random_pixels \
  --obs_fraction 0.05 \
  --split_strategy time_block \
  --out masks/splits_random_pixels_005_seed0.npz
```

This also writes an interactive HTML viewer next to the `.npz` file:

```text
masks/splits_random_pixels_005_seed0.html
```

Useful mask models:

```text
random_pixels
moving_particles
fixed_sensors
regular_grid
random_blocks
center_block
center_hole
temporal_stride
temporal_windows
```

Useful split strategies:

```text
time_block
random_observed
spatial_block
spatiotemporal_block
```

For block-style sparse observations:

```bash
python scripts/make_masks_splits.py \
  --u u_stack.npz \
  --v v_stack.npz \
  --crop_h 64 \
  --crop_w 64 \
  --mask_model random_blocks \
  --obs_fraction 0.05 \
  --block_h 8 \
  --block_w 8 \
  --split_strategy time_block \
  --out masks/splits_random_blocks_005_seed0.npz
```

To make a smaller HTML file without embedded velocity previews:

```bash
python scripts/make_masks_splits.py \
  --u u_stack.npz \
  --v v_stack.npz \
  --crop_h 64 \
  --crop_w 64 \
  --mask_model random_pixels \
  --obs_fraction 0.05 \
  --split_strategy time_block \
  --out masks/splits_random_pixels_005_seed0.npz \
  --no_data_preview
```

### Train On Dense Data

Run the original dense-supervision workflow:

```bash
python train.py \
  --u u_stack.npz \
  --v v_stack.npz \
  --crop_h 64 \
  --crop_w 64 \
  --model_type conv2d \
  --device cpu
```

### Train With A Mask/Split Artifact

Use a generated `.npz` mask file:

```bash
python train.py --u u_stack.npz --v v_stack.npz --crop_h 64 --crop_w 64 --model_type conv2d --device cpu --mask_npz masks/splits_random_pixels_005_seed0.npz
```

The training data loss uses `train_mask`. Validation and test data losses are
reported from `val_mask` and `test_mask`.

### Common Training Options

```bash
python train.py \
  --u u_stack.npz \
  --v v_stack.npz \
  --crop_h 64 \
  --crop_w 64 \
  --model_type mlp \
  --hidden 64,64,64 \
  --time_stride 20 \
  --n_physics 10000 \
  --adam_steps 1000 \
  --lbfgs_maxiter 1000 \
  --log_every 5 \
  --save_dir outputs
```

Model choices:

```text
mlp
conv2d
conv3d
```

### Outputs

Training writes figures to `--save_dir`, including:

```text
loss_history.png
split_data_loss_history.png
u_triptych.png
v_triptych.png
u_test_unmasked_triptych.png
v_test_unmasked_triptych.png
omega_triptych.png
quiver_true.png
quiver_pred.png
```

`u_triptych.png` and `v_triptych.png` show full-field ground truth, prediction,
and absolute error. The `*_test_unmasked_triptych.png` files show full-field
ground truth and prediction, plus the held-out test mask used for sparse
evaluation.
