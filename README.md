# Machine Learning Assignment 2

**Student:** Aman Kumar  
**Roll Number:** M26MAC001  
**Dataset:** Olivetti Faces  
**Random Seed:** 26001

## 1. Overview

This repository contains the Jupyter notebooks, reusable Python modules, saved models, numerical results, and figures for Machine Learning Assignment 2.

The experiments investigate decision trees, bagging, multiclass SAMME boosting, robustness to masked image regions, and empirical bias–variance analysis.

## 2. Dataset and Experimental Setup

- **Dataset:** Olivetti Faces, comprising 400 images across 40 identities.
- **Image representation:** Raw flattened 64 × 64 pixel arrays, without PCA or other dimensionality reduction.
- **Data split:** Six training images, two validation images, and two test images per identity.
- **Random seed:** 26001, derived from the numeric characters of roll number M26MAC001.
- **Robustness experiment:** A centered 16 × 16 region, covering rows and columns 24–39, is set to zero in the test images.
- **Training protocol:** Models are trained exclusively on clean images.
- **Model selection:** Validation accuracy is used to select model configurations before final test evaluation.
- **Ensemble methods:** Bagging and multiclass SAMME boosting are implemented in the project.
- **Bias–variance analysis:** Uses 30 resampled training sets with a fixed clean test set.

## 3. Repository Structure

```text
PythonProject3/
├── .venv/
├── Code/
│   ├── Q1.ipynb
│   ├── Q2.ipynb
│   ├── Q3.ipynb
│   └── Split.ipynb
├── common/
│   ├── Bagging.py
│   └── Boosting.py
├── Data/
│   ├── olivetti_py3.pkz
│   ├── X_test.npy.npz
│   ├── X_train.npy.npz
│   └── X_valid.npy.npz
├── Figures/
├── Models/
├── Result/
├── README.md
└── requirements.txt
```

### Directory Description

- `Code/` — Jupyter notebooks for the assignment experiments.
- `common/` — Reusable implementations of bagging and boosting.
- `Data/` — Dataset and saved training, validation, and test splits.
- `Figures/` — Generated plots and visualizations.
- `Models/` — Saved trained models.
- `Result/` — Numerical results and experiment outputs.
- `README.md` — Project documentation.
- `requirements.txt` — Python dependencies.