

# Diabetic Retinopathy Grading with Explanation Supervision

This repository contains code for a research project on **diabetic retinopathy (DR) severity grading** using deep learning, with a focus on **model interpretability and generalization**.
The core idea is to improve the reliability of model explanations by **aligning Grad-CAM heatmaps with expert-annotated lesion masks** during training.

---

## Motivation

Deep learning models for DR grading often achieve high accuracy but produce **unreliable or clinically implausible explanations**.
Additionally, models trained on one dataset frequently **fail to generalize** to images from other sources due to domain shift.

This project investigates whether **explicit supervision of explanations**, using lesion annotations, can:

* Encourage models to focus on clinically relevant regions
* Improve interpretability consistency
* Enhance cross-dataset generalization

---

## Key Contributions

* Implemented a **baseline DR classifier** using ResNet architectures
* Designed an **explanation alignment loss** to supervise Grad-CAM maps using lesion masks
* Trained models on **multiple DR datasets** with diverse imaging conditions
* Evaluated both **classification performance** and **explanation quality**
* Analyzed generalization under **cross-dataset testing**

---

## Datasets

The project uses a combination of publicly available DR datasets:

* EyePACS
* APTOS 2019
* Messidor
* IDRiD (for lesion supervision)
* DDR / E-Ophtha (where applicable)

Images are **not included** in this repository.
Please download datasets separately and organize them as described below.

---

## Directory Structure

```text
DR_PROJECT/
│
├── datasets/                 # Dataset loaders and preprocessing
├── models/                   # Model definitions (ResNet-based)
├── training/                 # Training and validation scripts
├── evaluation/               # Metrics, confusion matrices, analysis
├── explainability/           # Grad-CAM generation and visualization
├── losses/                   # Explanation alignment loss
├── utils/                    # Helper functions
├── experiments/              # Ablation and cross-dataset experiments
│
├── requirements.txt
└── README.md
```

---

## Method Overview

### Baseline Model (Model A)

* ResNet-based classifier trained using **image-level DR grades**
* Standard cross-entropy loss
* Data augmentation and normalization applied

### Explanation-Supervised Model (Model B)

* Generates Grad-CAM heatmaps during training
* Uses **lesion masks** to supervise explanations
* Optimizes a combined objective:

  * Classification loss
  * Explanation alignment loss (weighted by λ)

---

## Experiments

* In-domain evaluation
* Cross-dataset generalization testing
* Ablation studies with different alignment weights (λ)
* Qualitative visualization of Grad-CAM maps
* Comparison of baseline vs explanation-supervised models

---

## Key Findings

* Explanation supervision improves **spatial alignment** between model attention and lesion regions
* Classification accuracy gains are modest, but **interpretability consistency improves**
* Models with explanation supervision show **more stable behavior across datasets**
* Results highlight the importance of evaluating **beyond accuracy metrics**

---

## Requirements

```text
Python 3.9+
PyTorch
torchvision
numpy
pandas
opencv-python
matplotlib
scikit-learn
```

---

## Running the Code (Example)

```bash
python train_baseline.py
python train_explanation_supervised.py
```

(Adjust paths and dataset configuration as needed.)

---

## Research Status

* Baseline and explanation-supervised models implemented
* Experimental evaluation completed
* Manuscript preparation in progress

---

## Disclaimer

This code is intended for **research and educational purposes only** and is **not suitable for clinical deployment**.

---

## Contact

For questions or discussion related to this project, feel free to reach out.
