# 🤖 Project: Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs

This project focuses on building a deep learning-based diagnostic pipeline for classifying scintigraphic DICOM images of horse joints. It is structured into multiple steps, starting with a binary classification of **FTU (Functionally Targeted Uptake)** vs **Non-FTU** regions.

---

## 📌 Step 1: FTU vs Non-FTU Classification

This module implements a CNN-based classification system to distinguish between **FTU (Functionally Targeted Uptake - Leg joint region)** and **Non-FTU** regions in scintigraphic images of horse joints.

This is the **first step** in the broader project:
**"Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs"**

---

## 📁 Directory Structure

FTU vs Non FTU Classification
- `Dataset.py` – DICOM dataset loader with preprocessing and augmentation
- `Model.py` – FTUCNN architecture definition
- `Train.py` – Script for training the model
- `Test.py` – Script for running inference on test data
- `Evaluation.py` – Generates classification report and confusion matrix
- `model_FTU_nonftu.pth` – Saved trained model

---

## 🧠 Model Details

| Component       | Description                                         |
|----------------|-----------------------------------------------------|
| Architecture    | Custom CNN (`FTUCNN`) with BatchNorm               |
| Input Shape     | 128 × 128                                           |
| Optimizer       | AdamW                                              |
| Learning Rate   | 0.0001                                             |
| Epochs          | 10                                                 |
| Loss Function   | CrossEntropyLoss with class weights                |
| Regularization  | BatchNorm + Weight Decay (0.0001)                  |
| Augmentations   | FTU: Contrast jitter + rotation<br>Non-FTU: None   |

---

## 🧪 Results (Updated)

### ✅ Classification Report
- **Validation Accuracy:** `98.54%`
- **FTU Recall (Sensitivity):** `96.32%`
- **Non-FTU Specificity:** `98.92%`
- **FTU Precision:** `94%`
- **Non-FTU Precision:** `99%`

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Non-FTU   | 0.99      | 0.99   | 0.99     | 1115    |
| FTU       | 0.94      | 0.96   | 0.95     | 190     |
| **Accuracy** |         |        | **0.99** | **1305** |

#### 📉 Confusion Matrix

![Confusion Matrix](Images/FTU%20classification(cm).png)

#### 📋 Classification Report Screenshot

![Classification Report](Images/FTU%20classification%20report.png)

#### 📈 Training vs Validation Loss Plot

![Loss Plot](Images/train%20Val%20loss.png)
