# Mangrove Segmentation — Kaggle GPU Edition (v6)

A deep learning pipeline using a **Residual U-Net architecture** built with TensorFlow/Keras to segment mangrove ecosystems in the Sundarbans using multi-spectral Sentinel-2 satellite imagery and standard RGB drone imagery.

This project implements a highly optimized, RAM-flat streaming data generator designed specifically to circumvent multi-threaded GDAL thread-teardown crashes (`rasterio.errors.EnvError`) inside `tf.data` background threads.

---
## 📥 Model Download

The trained model (`mangrove_v6.keras` ~180 MB) is available via:

### Option 1: GitHub Releases (Recommended)
→ [Download Latest Release](https://github.com/aayushraman07/mangrove-segmentation/releases)

### Option 2: Direct Download
- `mangrove_v6.keras` → [Download Model](https://github.com/aayushraman07/mangrove-segmentation/releases/download/v1.0/mangrove_v6.keras)
- `mangrove_v6_norm_stats.npz` → [Download Stats](https://github.com/aayushraman07/mangrove-segmentation/releases/download/v1.0/mangrove_v6_norm_stats.npz)

**After downloading**, place both files in the project root.

---

## 📁 Repository Structure

* `mangrove_segmentation_v6.ipynb`: The core Jupyter Notebook containing the training setup, data pipeline, model architecture, and inference functions.
* `config/mangrove_v6_norm_stats.npz`: Cached global normalization stats (mean and standard deviation) computed across the Sundarbans dataset channels.
* `outputs/`:
    * `training_history.png`: Visual loss and Intersection over Union (IoU) curves across training epochs.
    * `predictions_preview.png`: Sample test predictions comparing true RGB layouts against generated ground truths and model predictions.
    * `training_log.csv`: Epoch-by-epoch evaluation tracking logs.

---

## 🛠️ Key Features & Technical Fixes

1. **Thread-Safe GDAL Pipeline**: Process-wide GDAL configurations are injected directly into environment variables instead of nesting thread-local `rasterio.Env()` instances, successfully preventing crashes across `tf.data` parallel workers.
2. **On-the-Fly Mask Generation**: Features a programmatic rule-based generator using precomputed **NDVI** and **NDMI** gates customized for coastal wetland vegetation to overcome unuseable/blank label channels.
3. **RAM-Flat Data Engine**: Utilizes a streaming generator factory that yields raw patch indices natively, dropping the shuffle buffer from massive pools down to 100 samples to save ~3.5GB of overhead RAM.
4. **Dual-Mode Inference Backend**: Shared sliding-window prediction matrices built to handle large multi-spectral GeoTIFFs (`.tif`) alongside simulated multi-band synthesized arrays derived from standard RGB drone video feeds.

---

## 📊 Model & Architecture

The framework relies on a modified **Residual U-Net** utilizing mixed-precision (`mixed_float16`) operations:

* **Inputs**: 8 channels (`B2, B3, B4, B8, B11, NDVI, NDWI, NDMI`)
* **Encoder Blocks**: Residual connections coupled with Batch Normalization and Dropout ($0.3$) at the bottleneck layer.
* **Decoder Blocks**: Bilinear upsampling components paired with concatenation skip-connections.
* **Loss Profile**: Combined Binary Cross-Entropy + Dice Loss function.

### Training Performance Preview

| Loss & Metric History | Model Predictions Example |
| :---: | :---: |
| ![Loss/IoU History](outputs/training_history.png) | ![Predictions Preview](outputs/predictions_preview.png) |

---

## 🚀 How To Run

Since the original dataset consists of multi-gigabyte geospatial satellite assets hosted on Kaggle, you can reproduce or evaluate this workflow by importing the notebook straight back into a Kaggle Environment.

### Setup Requirements
Ensure your workspace includes the following packages:
```bash
pip install tensorflow rasterio opencv-python scikit-learn matplotlib pandas numpy
