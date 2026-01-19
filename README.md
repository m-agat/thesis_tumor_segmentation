# UPC Master Thesis 2024 / 2025

> **Towards Reliable Brain Tumor Segmentation in MRI Neuroimaging**  
> **Integrating Uncertainty Estimation and Ensemble Methods for Clinical Applications**

This repository contains the code, data‑preparation steps, experiments, and web‑app for my UPC MSc thesis. I develop and evaluate an uncertainty‑aware ensemble of four state‑of‑the‑art 3D deep‑learning models for brain tumor segmentation in multimodal MRI, and integrate it into an interactive clinical web application.

---

## 🚀 Project Overview

Accurate brain tumor segmentation in MRI is essential for diagnosis, treatment planning, and monitoring. However, deep‑learning models can be over‑confident and fail silently - especially on rare tumor subregions or out‑of‑distribution scans.

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
    ├── brain_seg_app                ← Streamlit web application
    ├── calibration                  ← Probability calibration routines
    ├── config                       ← Config files
    ├── dataset                      ← Data loaders & preprocessing
    ├── ensemble                     ← Ensemble‑fusion code
    ├── models                       ← Model definitions (V‑Net, SegResNet, etc.)
    ├── ood_samples                  ← Out‑of‑distribution test cases
    ├── stats                        ← Statistical analysis scripts
    ├── train                        ← Training & hyperparameter tuning scripts
    ├── uncertainty                  ← Uncertainty estimation methods and evaluation
    ├── utils                        ← Utility functions
    └── visualization                ← Plotting & figure generation
```

## 📈 Results
Dice scores for each model:

![Dice scores](assets/dice_scores_indiv_vs_ensemble.png)

Full analysis of results can be found in the project report. 

## 🎬 Web App Demo
Below is a quick demo of the interactive segmentation interface:

1. **Upload your MRI scans**
   
![File upload](assets/upload_files.gif)

3. **Run the segmentation and uncertainty estimation**

![Running segmentation](assets/run_segmentation.gif)
   
5. **Explore results slice‑by‑slice and download them if you want**

![Showing results](assets/show_results.gif)
