# Understanding Where Accuracy Is Lost Under Visual Degradation: A Systematic Error Decomposition Study

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-62%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical computer vision research framework investigating **why** object detectors fail under adverse visual conditions, rather than reporting only aggregate performance drops such as mAP.

---

## 1. Research Motivation & Core Question

Standard robustness benchmarks report aggregate summary statistics (e.g., mAP drop from $0.52 \rightarrow 0.22$). However, identical drops in aggregate mAP can arise from fundamentally different failure modes:
- A detector may **miss objects entirely** (False Negatives / Object Loss).
- It may **localize objects poorly** (bounding box boundary slips).
- It may **confuse semantic classes** (classification error).
- It may **generate duplicate predictions** (NMS redundancy).
- It may **hallucinate objects on background artifacts** (Background False Positives).

This research project performs a systematic diagnostic error decomposition of modern object detectors across degradation types, severity levels, and detector architectures.

### Central Research Questions
1. **Failure Mechanism Shift**: *Does the primary failure mechanism change with the type and severity of visual degradation?*
2. **Cross-Architecture Discrepancy**: *Do Transformer-based detectors fail differently from one-stage and two-stage CNN architectures?*
3. **Restoration Trade-off Hypothesis**: *Does visual restoration (e.g., low-light enhancement) recover missed detections at the cost of increasing background false positives due to amplified artifacts?*

---

## 2. Diagnostic Error Taxonomy

Every ground-truth object and detector prediction is partitioned into mutually exclusive, mathematically conserved categories:

| Error Category | Diagnostic Definition | Primary Cause |
|---|---|---|
| **True Positive (TP)** | Correct class, $\text{IoU} \ge \tau_{\text{pos}}$ ($0.50$), primary ground truth claim | Accurate detection |
| **Missed Detection (FN)** | Ground-truth object with no valid primary positive detection | Severe illumination loss, extreme blur, occlusion |
| **Background FP (Bkg)** | Detection with $\text{IoU} < \tau_{\text{loc}}$ ($0.10$) with all ground truths | Sensor noise, edge artifacts, visual hallucination |
| **Localization Error (Loc)** | Correct class, but $\tau_{\text{loc}} \le \text{IoU} < \tau_{\text{pos}}$ ($0.10 \le \text{IoU} < 0.50$) | Spatial detail loss, blur, boundary softening |
| **Classification Error (Cls)** | Overlaps another ground truth ($\text{IoU} \ge 0.50$), but predicted wrong category | Texture loss, contrast degradation, color distortion |
| **Class + Loc Error (Cls+Loc)** | Wrong category and moderate localization ($\tau_{\text{loc}} \le \text{IoU} < 0.50$) | Severe corruption combining shape and texture loss |
| **Duplicate Detection (Dupe)** | Correct class, $\text{IoU} \ge 0.50$, but GT was already claimed by a higher-confidence box | Non-Maximum Suppression (NMS) failure |

### Exact Conservation Laws
$$\text{Total Detections} = N_{\text{TP}} + N_{\text{Dupe}} + N_{\text{Loc}} + N_{\text{Cls}} + N_{\text{Cls\_Loc}} + N_{\text{Bkg}}$$
$$\text{Total Ground Truth} = N_{\text{TP}} + N_{\text{Miss}}$$

---

## 3. Supported Experimental Matrix

### 3.1 Three Detector Families
- **One-Stage CNN**: YOLO11 (`yolo11n.pt` via Ultralytics)
- **Two-Stage Anchor-Based CNN**: Faster R-CNN (`fasterrcnn_resnet50_fpn` via Torchvision)
- **Transformer-Based**: RT-DETR (`rtdetr-l.pt` via Ultralytics)

### 3.2 Five Calibrated Degradations (Severities 1 to 5)
1. **Low Light** (`low_light`): Non-linear gamma curve ($\gamma \in [1.3, 3.2]$) + luminance attenuation ($\beta \in [0.80, 0.12]$).
2. **Gaussian Blur** (`gaussian_blur`): Defocus filtering with increasing kernel standard deviation ($\sigma \in \{1.0, 2.0, 3.5, 5.0, 7.0\}$).
3. **Gaussian Noise** (`gaussian_noise`): Zero-mean additive thermal sensor noise ($\sigma \in \{8, 16, 28, 45, 70\}$ intensity units).
4. **Fog / Haze** (`fog`): Atmospheric scattering model ($I = J \cdot t + A(1-t)$) with transmission $t \in \{0.80, 0.65, 0.50, 0.35, 0.20\}$.
5. **JPEG Compression** (`jpeg_compression`): Lossy block DCT quantization artifacts ($Q \in \{60, 40, 25, 15, 7\}$).

### 3.3 Image Restoration Methods & Trade-off Evaluation
- **Low-Light Enhancement**: CLAHE on LAB luminance, Gamma correction, Single-scale Retinex.
- **Dehazing**: Dark Channel Prior (DCP) transmission inversion ($J = \frac{I - A}{\max(t, t_0)} + A$), localized CLAHE.
- **Denoising**: Bilateral edge-preserving filtering, Gaussian smoothing, Median filtering.
- **Triad Study**: Evaluates Clean $\rightarrow$ Degraded $\rightarrow$ Restored to quantify recovered misses vs. introduced hallucinations.

---

## 4. Repository Structure

```text
major_project/
├── config.yaml               # Unified model and inference hyperparameters
├── requirements.txt          # Pinned project dependencies
├── run_experiments.py        # Master research experimental suite driver
├── pytest.ini                # Pytest configuration
├── src/                      # Core research package
│   ├── __init__.py
│   ├── detect.py             # Multi-architecture detector runner (YOLO, Faster R-CNN, RT-DETR)
│   ├── evaluate.py           # Standard COCO mAP evaluation via pycocotools
│   ├── degradations.py       # Image degradation engine (5 types x 5 severities)
│   ├── error_analysis.py     # Systematic diagnostic error decomposition engine
│   ├── restoration.py        # Image restoration algorithms & trade-off evaluation
│   ├── pipeline.py           # Clean vs. degraded experiment orchestrator
│   ├── visualize.py          # Research figures (stacked bar charts, trajectories) & LaTeX tables
│   └── prepare_sample_data.py# Offline COCO-compliant synthetic benchmark generator
├── scripts/                  # Shell execution scripts
│   └── run_val2017.sh
├── results/preliminary/      # Archived preliminary detection results
│   ├── yolo11_50.json
│   ├── fasterrcnn_50.json
│   └── rtdetr_50.json
└── tests/                    # Comprehensive test suite (62 tests passing)
    ├── test_detect_utils.py
    ├── test_degradations.py
    ├── test_error_analysis.py
    ├── test_restoration.py
    ├── test_pipeline.py
    ├── test_visualize.py
    └── test_run_experiments.py
```

---

## 5. Installation & Quickstart

### 5.1 Environment Setup
```bash
# Clone the repository
git clone git@github.com:dagaayush1205/major_project.git
cd major_project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Verification in 10 Seconds (Dry-Run / CPU Mode)
Run the entire experimental matrix with simulated detection and restoration trade-offs:
```bash
python run_experiments.py --dry-run --restore --out-dir sample_results
```
This automatically:
- Generates a synthetic COCO sample benchmark in `sample_benchmark/`.
- Executes clean vs. degraded conditions across all 3 models and 5 degradations.
- Performs diagnostic error decomposition and restoration trade-off studies.
- Saves publication plots in `sample_results/figures/` and thesis tables in `sample_results/tables/`.

### 5.3 Run the Test Suite
```bash
python3 -m pytest tests/ -v
```
All 62 unit and integration tests validate monotonicity, conservation laws, format preservation, and end-to-end pipeline execution.

---

## 6. Running Live Research Experiments

### 6.1 Running on COCO or Custom Dataset
Ensure clean images are placed in `images/` and annotations in `annotations/instances_val.json`:
```bash
python run_experiments.py \
  --images images/ \
  --gt annotations/instances_val.json \
  --models yolo11 fasterrcnn rtdetr \
  --degradations low_light gaussian_blur gaussian_noise fog jpeg_compression \
  --severities 1 2 3 4 5 \
  --restore \
  --out-dir results_live/
```

### 6.2 Running an Isolated Degradation Experiment
```bash
python pipeline.py \
  --images sample_data/images \
  --gt sample_data/annotations/instances_sample.json \
  --model yolo11 \
  --degradation low_light \
  --severities 1 2 3 4 5 \
  --out-dir results_yolo_lowlight
```

### 6.3 Evaluating Restoration Trade-offs Directly
```bash
python -c "
from restoration import run_restoration_tradeoff_study
study = run_restoration_tradeoff_study(
    clean_images_dir='sample_data/images',
    gt_source='sample_data/annotations/instances_sample.json',
    model_name='yolo11',
    degradation_type='low_light',
    severity=3,
    restoration_type='low_light_enhancement'
)
print(study['tradeoff_diagnosis'])
"
```

### 6.4 Exporting Figures & Thesis Tables
```bash
python visualize.py \
  --input results_yolo_lowlight/yolo11_low_light_summary.json \
  --out-dir thesis_assets/
```
Outputs:
- `thesis_assets/yolo11_low_light_error_breakdown.png`
- `thesis_assets/yolo11_low_light_trajectories.png`
- `thesis_assets/yolo11_low_light_table.csv`
- `thesis_assets/yolo11_low_light_table.md`
- `thesis_assets/yolo11_low_light_table.tex` (LaTeX code ready for inclusion in thesis document)

---

## 7. Citation & Major Project Information
- **Title**: Understanding Where Accuracy Is Lost Under Visual Degradation
- **Focus**: Systematic Error Decomposition of Object Detectors & Restoration Trade-offs
- **Authors**: Major Project Research Team
