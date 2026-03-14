# 🌿 Deep Learning-Based Weed Detection for Precision Agriculture
### Benchmarking Single-Stage and Multi-Stage Detectors with Re-parameterized Convolutional Layer (RCL) Enhancement

> A deep learning pipeline that evaluates five state-of-the-art object detection models for weed detection in UAV agricultural imagery — benchmarking YOLO XL, InternImage-H, Co-DETR, MaxViT-B, and Cascade Eff-B7 NAS-FPN, then enhancing the top three with a Re-parameterized Convolutional Layer (RCL) to push detection accuracy further.

---

## 📌 Overview

Weeds compete with crops for essential resources, yet most detection systems sacrifice either speed or accuracy to manage them. This project asks a more structured question: which state-of-the-art deep learning architecture best balances detection precision and inference speed in complex UAV agricultural imagery — and how much further can a targeted architectural modification push that performance?

Using high-resolution annotated UAV images from Mendeley Data, this project benchmarks two single-stage and three multi-stage detectors across precision, recall, F1 score, and accuracy. The top three models are then enhanced with a Re-parameterized Convolutional Layer (RCL) — a novel modification designed to improve spatial feature extraction — and evaluated before and after integration.

| Goal | Approach |
|------|----------|
| Benchmark leading detection models on weed imagery | 5 models evaluated: YOLO XL, InternImage-H, Co-DETR, MaxViT-B, Cascade Eff-B7 NAS-FPN |
| Compare single-stage vs. multi-stage detection strategies | Speed vs. accuracy tradeoff analysis across architectures |
| Enhance feature extraction in top models | Re-parameterized Convolutional Layer (RCL) integrated into YOLO XL, InternImage-H, Co-DETR |

---

## 📂 Dataset

**Deep Learning-Based Early Weed Segmentation Using Motion Blurred UAV Images of Sorghum Fields**
Source: [Mendeley Data](https://data.mendeley.com/datasets/4hh45vkp38/5)

- **Framework base:** UAVWeedSegmentation
- **Image type:** High-resolution UAV imagery of sorghum fields under varied lighting and environmental conditions
- **Annotations:** Pixel-level masks distinguishing weed species from crop plants
- **Splits:** Structured train / validation / test partitions for rigorous evaluation

### Dataset Characteristics

| Property | Description |
|-----------|-------------|
| **Source** | Aerial UAV imagery, sorghum agricultural fields |
| **Conditions** | Variable lighting, weather, and soil backgrounds |
| **Annotation type** | Segmentation masks per image |
| **Task** | Binary weed vs. crop detection and segmentation |

---

## 🔧 Tech Stack

| Category | Libraries / Tools |
|----------|-------------------|
| Deep Learning Framework | `PyTorch`, `torch.nn` |
| Model Architectures | `YOLOX`, `InternImage`, `Co-DETR`, `MaxViT`, `EfficientNet + NAS-FPN` |
| RCL Implementation | `ReParameterizedConv2d` (custom PyTorch module) |
| Data Processing | `numpy`, `pandas`, `scikit-learn`, `kornia` |
| Image I/O | `skimage`, `PIL` |
| Visualization | `matplotlib`, `torchviz` |
| Training Utilities | `UAVWeedSegmentation` (custom framework), `optuna` |
| Evaluation | `sklearn.metrics` (confusion matrix, classification report) |

---

## 🗂️ Repository Structure

```
📦 weed-detection-precision-agriculture
 ┣ 📜 weed_yoloxl.ipynb                    # YOLO XL baseline — training, inference, evaluation
 ┣ 📜 weed_yoloxl-RCL.ipynb               # YOLO XL with RCL enhancement
 ┣ 📜 weed_co_detr.ipynb                   # Co-DETR baseline — training, inference, evaluation
 ┣ 📜 weed_co_detr_RCL.ipynb              # Co-DETR with RCL enhancement
 ┣ 📜 weed_internlmage_h.ipynb             # InternImage-H baseline — training, inference, evaluation
 ┣ 📜 weed_internlmage_h_RCL.ipynb        # InternImage-H with RCL enhancement
 ┣ 📜 weed_maxvit.ipynb                    # MaxViT-B baseline — training, inference, evaluation
 ┣ 📜 weed_cascade.ipynb                   # Cascade Eff-B7 NAS-FPN — training, inference, evaluation
 ┗ 📜 README.md
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- Downloaded dataset from Mendeley Data and unzipped into structured `train/val/test` directories
- Images saved as patches using `save_patches.py` to prepare fixed-size inputs for model training
- Per-fold mean and standard deviation calculated for normalization across K-fold cross-validation splits
- Custom data loaders built using `kornia` for GPU-accelerated augmentation

### 2. Model Architectures

Five detection models were implemented and trained — two single-stage detectors for speed, three multi-stage detectors for precision:

**Single-Stage Detectors** process the full image in one forward pass, predicting bounding boxes and class labels simultaneously — ideal for real-time applications:

| Model | Key Design |
|-------|-----------|
| **YOLO XL** | YOLOX-L backbone with FPN blocks (P3–P5) and a decoupled detection head |
| **InternImage-H** | Hierarchical multi-scale architecture with downsampling stages and LN + FCN detection head |

**Multi-Stage Detectors** generate region proposals first, then refine and classify — trading speed for precision:

| Model | Key Design |
|-------|-----------|
| **Co-DETR** | ResNet50 CNN backbone + Transformer encoder-decoder with positional encoding |
| **MaxViT-B** | Multi-axis Vision Transformer with MBConv blocks, block attention, and grid attention |
| **Cascade Eff-B7 NAS-FPN** | EfficientNet-B7 with NAS-optimized Feature Pyramid Network for multi-scale detection |

### 3. Training and Evaluation

All models trained using K-fold cross-validation with `optuna`-based hyperparameter optimization. Evaluated on a held-out test set using four metrics:

```python
# Evaluation via scikit-learn classification report
from sklearn.metrics import confusion_matrix, classification_report

report = classification_report(ground_truth, predictions)
cm = confusion_matrix(ground_truth, predictions)
```

| Metric | What It Measures |
|--------|-----------------|
| **Precision** | Ratio of true weed detections to all detections |
| **Recall** | Ratio of true weed detections to all actual weeds |
| **F1 Score** | Harmonic mean of precision and recall |
| **Accuracy Score** | Overall correct predictions across all classes |

### 4. Re-parameterized Convolutional Layer (RCL)

After benchmarking all five models, the top three (YOLO XL, InternImage-H, Co-DETR) were enhanced with a custom `ReParameterizedConv2d` module — a probabilistic convolutional layer that learns weight distributions rather than fixed weights, enabling more adaptive spatial feature extraction:

```python
class ReParameterizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ReParameterizedConv2d, self).__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels,
                                                    kernel_size, kernel_size))
        # Learnable mean and variance for re-parameterized sampling
        ...
```

> **Key Insight:** By re-parameterizing convolutional operations, the RCL enables more adaptive learning of spatial hierarchies — critical for distinguishing subtle weed structures against complex agricultural backgrounds.

---

## 📊 Key Results

**Baseline Model Comparison**

| Model | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|--------|----------|----------|
| YOLO XL | 0.85 | 0.72 | 0.785 | 98.21% |
| InternImage-H | 0.8345 | 0.7456 | 0.7901 | 96.21% |
| Co-DETR | 0.80 | 0.70 | 0.750 | 90.21% |
| Cascade Eff-B7 NAS-FPN | 0.7543 | 0.7012 | 0.7278 | 89.21% |
| MaxViT-B | 0.7845 | 0.7123 | 0.7484 | 89.01% |

**Top 3 Models — Before vs. After RCL Integration**

| Model | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|--------|----------|----------|
| YOLO XL (baseline) | 0.85 | 0.72 | 0.785 | 98.21% |
| **YOLO XL + RCL** | **0.87** | **0.80** | **0.835** | **98.89%** |
| InternImage-H (baseline) | 0.8345 | 0.7456 | 0.7901 | 96.21% |
| InternImage-H + RCL | 0.80 | 0.71 | 0.755 | 98.21% |
| Co-DETR (baseline) | 0.80 | 0.70 | 0.750 | 90.21% |
| **Co-DETR + RCL** | **0.83** | **0.80** | **0.815** | **93.24%** |

- YOLO XL with RCL achieved the highest overall accuracy (98.89%) and the greatest improvement in Recall (+0.08), confirming RCL's effectiveness for real-time single-stage detection
- Co-DETR showed consistent gains across all metrics post-RCL, demonstrating the layer's compatibility with Transformer-based multi-stage architectures
- InternImage-H showed a slight precision/recall tradeoff after RCL integration, indicating overfitting sensitivity — overall accuracy still improved to 98.21%
- Cascade Eff-B7 NAS-FPN and MaxViT-B maintained faster execution times but lower accuracy, highlighting a clear speed-accuracy tradeoff among multi-stage detectors

---

## ⚠️ Limitations

- InternImage-H showed signs of overfitting after RCL integration — the additional complexity may require model-specific regularization or learning rate tuning
- RCL integration consistently increased execution time across all models; YOLO XL's time rose from 2.0s to 3.5s per image, requiring careful consideration for real-time deployment
- The dataset captures a specific agricultural setting (sorghum fields) — generalization to other crops or environmental conditions has not been validated
- Models were trained and evaluated on static UAV snapshots; performance under dynamic field conditions (varying altitude, motion blur, seasonal change) remains an open question

---

## 🔮 Future Work

- **RCL parameter optimization** — tune regularization and learning rates specifically for InternImage-H to resolve overfitting while retaining accuracy gains
- **IoT and edge deployment** — integrate optimized models with IoT-based spraying systems for real-time in-field weed control
- **Broader agricultural tasks** — extend the pipeline to pest detection, disease identification, and crop stress monitoring
- **Field validation** — test models under real-world conditions across diverse crop types, altitudes, and seasonal variation
- **Interdisciplinary integration** — collaborate with agronomy and robotics to build end-to-end precision agriculture systems grounded in field practicalities

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/madhusomethingg/weed-detection-precision-agriculture.git
cd weed-detection-precision-agriculture

# Install dependencies
pip install -r requirements.txt
```

1. Download the dataset from [Mendeley Data](https://data.mendeley.com/datasets/4hh45vkp38/5)
2. Open the relevant notebook and run all cells top to bottom:
   - `weed_yoloxl.ipynb` — YOLO XL baseline
   - `weed_yoloxl-RCL.ipynb` — YOLO XL with RCL enhancement
   - `weed_co_detr.ipynb` / `weed_co_detr_RCL.ipynb` — Co-DETR baseline and enhanced
   - `weed_internlmage_h.ipynb` / `weed_internlmage_h_RCL.ipynb` — InternImage-H baseline and enhanced
   - `weed_maxvit.ipynb` — MaxViT-B baseline
   - `weed_cascade.ipynb` — Cascade Eff-B7 NAS-FPN baseline
3. Each notebook handles its own data loading, model definition, training, inference, and metric evaluation end-to-end

---


## 👤 Author

Madhumitha Rajagopal

---

## 📄 License

This project is for educational and research purposes.
