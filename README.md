# UPC Master Thesis 2024 / 2025

> **Towards Reliable Brain Tumor Segmentation in MRI Neuroimaging**  
> **Integrating Uncertainty Estimation and Ensemble Methods for Clinical Applications**

This repository contains the code, data‑preparation steps, experiments and web‑app for my UPC MSc thesis. We develop and evaluate an uncertainty‑aware ensemble of four state‑of‑the‑art 3D deep‑learning models for brain tumor segmentation in multimodal MRI, and integrate it into an interactive clinical web application.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)  
2. [Folder Structure](#folder-structure)  
3. [Getting Started](#getting-started)  
   - [Requirements](#requirements)  
   - [Installation](#installation)  
4. [Usage](#usage)  
   - [Training](#training)  
   - [Inference](#inference)  
   - [Web App](#web-app)  
5. [Results & Figures](#results--figures)  
6. [Citing This Work](#citing-this-work)  
7. [License](#license)  

---

## 🚀 Project Overview

Accurate brain tumor segmentation in MRI is essential for diagnosis, treatment planning, and monitoring. However, deep‑learning models can be over‑confident and fail silently—especially on rare tumor subregions or out‑of‑distribution scans.

**Contributions:**  
- Trained four 3D CNN/Transformer architectures (V‑Net, SegResNet, Attention U‑Net, SwinUNETR) via 5‑fold cross‑validation  
- Fused them with three ensemble strategies:  
  1. Simple averaging  
  2. Performance‑weighted averaging  
  3. Performance + uncertainty‑weighted averaging  
- Incorporated voxel‑wise uncertainty estimation (epistemic & aleatoric)  
- Wrapped everything into an interactive web app for clinical review  

---

## 📂 Folder Structure
```bash
├── EDA                              ← Exploratory data analysis notebooks
├── hyperparameter_tuning_results    ← Hyperparameter search outputs (logs, plots)
├── other                            ← Miscellaneous scripts & notes
└── src                              ← All source code
    ├── brain_seg_app                ← Flask/FastAPI web application
    ├── calibration                  ← Probability calibration routines
    ├── config                       ← YAML/JSON config files
    ├── confusion_matrices           ← Auto‑generated confusion matrices
    ├── dataset                      ← Data loaders & preprocessing
    ├── ensemble                     ← Ensemble‑fusion code
    ├── models                       ← Model definitions (V‑Net, U‑Net, etc.)
    ├── ood_samples                  ← Out‑of‑distribution test cases
    ├── stats                        ← Statistical analysis scripts
    ├── train                        ← Training & cross‑validation loops
    ├── uncertainty                  ← Uncertainty estimation methods
    ├── utils                        ← Utility functions
    └── visualization                ← Plotting & figure generation
```
