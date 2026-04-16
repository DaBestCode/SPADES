📦 SPADES Challenge – Training & Inference Repository

This repository contains the full pipeline for training and evaluating a deep learning model for the SPADES pose estimation challenge. It includes a highly optimized event-to-image representation, domain-adaptive augmentation strategies, and a dual-branch architecture for translation and rotation prediction.

🚀 Training Pipeline

The training pipeline is designed to bridge the domain gap between synthetic event data and real-world sensor observations.

1. Event Stream → Image Representation

Raw event data (x, y, t, polarity) is aggregated within a temporal window.
Events are projected into a 3-channel tensor (time-sliced accumulation).
Spatial sharpening, logarithmic scaling, and normalization are applied to produce a dense, learnable representation.

2. Dual-View Input Construction

Global view (img_f): Full-frame spatial context.
Local view (img_r): Cropped region centered around event density (object-focused).
A scale hint is computed to guide translation estimation.

3. Domain-Adaptive Augmentation Pipeline

Always-on physical effects:
Blue floor bias (sensor baseline)
Edge brightening (lighting effects)
Organic sensor noise
Vignetting and chromatic aberration
JPEG compression artifacts
Random augmentations (train only):
Motion blur, structural debris
Lens flare, background streaks
Secondary lighting effects

These augmentations simulate real sensor conditions and significantly improve generalization.

4. Model Architecture

Translation branch: EfficientNet-V2-S backbone
Rotation branch: ResNet-50 + CBAM attention
Outputs:
3D translation vector
Quaternion rotation (normalized)

5. Loss Function

Translation: Smooth L1 (Huber) loss
Rotation: Geodesic quaternion loss
Weighted combination emphasizes rotation learning.

6. Training Strategy

Mixed precision (AMP) for efficiency
Gradient clipping for stability
Dynamic learning rate schedule
Automatic checkpointing and resume support
🧪 Testing / Inference Pipeline

The inference pipeline is optimized for robustness and clean predictions on real test data.

1. Event Filtering & Preprocessing

Removal of background noise using multi-scale density masking
Hot-pixel suppression via histogram-based filtering

2. Event → Image Conversion

Same 3-channel tensor generation as training (ensures consistency)

3. Dual-Pass Inference

Global pass: Full-frame translation estimation
Local pass: Cropped object region for rotation refinement

4. Clean Inference Filters

Background clutter suppression
Conditional blur (applied when object dominates frame)
Mild blue-floor normalization for domain alignment

5. Model Prediction

Forward pass through trained network
Outputs:
Translation (Tx, Ty, Tz)
Rotation quaternion (Qx, Qy, Qz, Qw)

6. Submission Generation

Predictions are written into a CSV file following challenge format
Includes fallback handling for edge cases (low event density, errors)
📊 Pipeline Visualization

The repository also includes a pipeline visualization notebook (pipeline_viz.ipynb) that illustrates:

Event-to-image transformation steps
Augmentation effects (before vs after)
Dual-branch model flow
End-to-end training and inference pipeline

This file serves as a visual guide to better understand how raw event data is processed into final pose predictions.
