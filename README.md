# Master Thesis 2024

## **Automated Tumor Segmentation in MRI Neuroimaging - From Advanced AI techniques to Clinical Toolbox**

# **Master Thesis 2024: Automated Tumor Segmentation in MRI Neuroimaging**

## **Project Title**: From Advanced AI Techniques to Clinical Toolbox

---

### **Project Overview**

This project focuses on developing an AI-driven tumor segmentation tool using MRI neuroimaging. The objective is to apply deep learning techniques to automate tumor detection and segmentation, aiming to create a clinical toolbox to assist in neuroimaging analysis for brain tumors.

---

### **Pipeline Overview**

1. **Step 1: Preprocessing** (Completed)
    - Includes skull stripping, normalization, and resizing MRI images for uniformity.
    
2. **Step 2: Model Training & Validation** (In Progress)
    - Testing different models like 3D UNet, VNet, SegResNet, ResUNet++, and Attention UNet.
    
3. **Step 3: Model Evaluation & Optimization** (To Do)
    - Evaluation using Dice scores and optimization for clinical application.

---

### **Data Structure**

To run the project, organize the **BraTS 2024 Dataset** in the following folder structure under the `data/` directory:

```plaintext
BraTS2024-BraTS-GLI-TrainingData
└── training_data1_v2
    ├── BraTS-GLI-00005-100
    │   ├── BraTS-GLI-00005-100-seg.nii.gz
    │   ├── BraTS-GLI-00005-100-t1c.nii.gz
    │   ├── BraTS-GLI-00005-100-t1n.nii.gz
    │   ├── BraTS-GLI-00005-100-t2f.nii.gz
    │   └── BraTS-GLI-00005-100-t2w.nii.gz
    ├── BraTS-GLI-00005-101
    └── ...
    
BraTS2024-BraTS-GLI-ValidationData
└── validation_data
    ├── BraTS-GLI-02073-100
    │   ├── BraTS-GLI-02073-100-t1c.nii.gz
    │   ├── BraTS-GLI-02073-100-t1n.nii.gz
    │   ├── BraTS-GLI-02073-100-t2f.nii.gz
    │   └── BraTS-GLI-02073-100-t2w.nii.gz
    └── ...
