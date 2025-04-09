---
layout: single
title: Foundation Model
permalink: /projects/deeplense/
author_profile: false
---

## **Specific Test VI - Foundation Model**  

This folder contains my solution for **Specific Test VI: Foundation Model** of the DeepLense GSoC 2025 project. The task involves pretraining a **Masked Autoencoder (MAE)** on strong lensing images and fine-tuning it for **multi-class classification** and **super-resolution** using **PyTorch**.  

### 📌 **Task Overview**  
The test consists of two main parts:  
1. **Pretraining a Masked Autoencoder (MAE)** on **no_sub** samples to learn meaningful feature representations.  
2. **Fine-tuning the MAE**:  
   - For **multi-class classification** (distinguishing between no_sub, cdm, and axion).  
   - For **super-resolution** (upscaling low-resolution images using high-resolution ground truths).  

#### 📷 Sample Images for Each Task
- **Samples for multi-class classification**
   ![Sample Images](/assets/deeplense/classification/classSample.png)
- **Samples for super-resolution**
   ![Sample Images](/assets/deeplense/superresolution/superRsample.png)


### **Prepate Data for Masked Autoencoder (MAE) Pretraining**  

#### **Input for Encoder**
- **Sample for splitted-image**
   ![Sample Images](/assets/deeplense/mae/splitted_image.png)

- **Sample for masked-image**
   ![Sample Images](/assets/deeplense/mae/masked_image.png)

- **Masked pathces and Visible patches**

   - ![Masked Image](/assets/deeplense/mae/masked_patches.png)
   - ![EncoderInput](/assets/deeplense/mae/visible_patches.png)

### 🛠 **Model and Approach**  
#### **1️⃣ Masked Autoencoder (MAE) Pretraining**
- **Goal:** Learn a feature representation of strong lensing images.  
- **Architecture:** Vision Transformer (ViT) backbone with a reconstruction head.  
- **Pretraining Loss:** Mean Squared Error (MSE)
- **Optimizer:** AdamW 
- **Batch Size:** *256*
- **Epochs:** *250*

#### **2️⃣ Fine-Tuning for Multi-Class Classification**
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** AdamW 
- **Batch Size:** *256*
- **Evaluation Metrics:** AUC Score, Accuracy  
- **Epochs:** *250*

#### **3️⃣ Fine-Tuning for Super-Resolution**
- **Loss Function:** Mean Squared Error (MSE)
- **Batch Size:** *256*
- **Evaluation Metrics:** MSE, SSIM, PSNR
- **Epochs:** *200*
- **NOTE** The Decoder used here is not suitable for images and especially for super-resolution tasks. So we need more work on the architecture

### 📊 **Results**  
Below are the evaluation results for each task:  

#### **1️⃣ Masked Autoencoder (MAE) Pretraining**  
- **Training Loss (MSE) over 250 epochs**  
  ![MAE Loss](/assets/deeplense/mae/MAE_Losses.png)  
- **PCA and TSNE on the embedding**  
  - Hidder representation:  
    - ![pca](/assets/deeplense/mae/pca_plot.png)
    - ![tsne](/assets/deeplense/mae/tsne_plot.png)

#### **2️⃣ Multi-Class Classification**  
- **Accuracy & AUC Score over epochs**  
    - ![Accuracy Metrics](/assets/deeplense/classification/Accuracies.png)
    - ![AUC Metrics](/assets/deeplense/classification/AUC.png)

- **Classification Report**
```
              precision    recall  f1-score   support

      no_sub       0.97      0.99      0.98      2945
       axion       0.98      0.97      0.97      2990
         cdm       0.97      0.95      0.96      2976

    accuracy                           0.97      8911
   macro avg       0.97      0.97      0.97      8911
weighted avg       0.97      0.97      0.97      8911
```

- **Confusion Matrix and ROC Curve**
    - ![ROC Metrics](/assets/deeplense/classification/ROC_curve.png)
    - ![Confusion Matrix](/assets/deeplense/classification/confusion_matrix.png)

- **PCA & tsne plotting**
    - ![PCA](/assets/deeplense/classification/pca_plot.png)
    - ![tsne](/assets/deeplense/classification/tsne_plot.png)

#### **3️⃣ Super-Resolution**
- **MSE as a loss, SSIM, PSNR over epochs**
      - ![SSIM](/assets/deeplense/superresolution/SSIM.png)
      - ![PSNR](/assets/deeplense/superresolution/PSNR.png)
      - ![MSE](/assets/deeplense/superresolution/MAE_Losses.png)

- **Final Metrics** *these results from best SSIM model **superresolution_SSIM** and it's very close to **superresolution_PSNR***
     - Final Validation MSE: 0.002293
     - Final Validation PSNR: 29.62
     - Final Validation SSIM: 0.9190
##### **Interpretation**

- Lower **MSE** means better reconstruction (less error).
- If MSE = 0, the images are identical.

- Higher **PSNR** means better quality.
- **Typical values:**
  - **30-50 dB** → Good quality
  - **20-30 dB** → Moderate quality
  - **<20 dB** → Poor quality
- If PSNR → ∞, it means the images are **identical** (MSE = 0).

- **SSIM = 1** → Identical images.
- **SSIM close to 0** → No structural similarity.
- Unlike MSE and PSNR, **SSIM aligns more with human perception**.

- **Super-resolution comparison**  
  - Low-res, predicted high-res, and ground truth  
    | ![LR](/assets/deeplense/superresolution/lr_image.png) | ![Predicted](/assets/deeplense/superresolution/superResoluted.png) | ![HR](/assets/deeplense/superresolution/hr_image.png) |