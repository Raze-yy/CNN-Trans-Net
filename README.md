# 🌈 Donaldson Matrix Prediction using CNN - Transformer Model

This repository provides PyTorch implementations for predicting excitation-emission matrices (EEMs) of fluorescent materials using hybrid CNN and Transformer models. The model takes incident and emission spectra as input and predicts a full 2D EEM.



---

## 📁 Project Structure

.
├── models/ # All model definitions (CNN, Transformer variants)
├── utils/ # Data loading and preprocessing utilities
├── data/ # Sample input and groundtruth data
├── train.py # Main training script
├── plot_loss.py # Script to visualize training/validation loss
├── requirements.txt # Python dependencies
└── README.md # This file


---

## 🚀 Getting Started

### 1. Install dependencies

```

pip install -r requirements.txt
📁 Data Description
This repository includes only a sample dataset for Sample 1 to demonstrate input/output data structure.

Folder structure:

data/
├── input/
│   └── 1.xlsx                  # Input excitation spectrum (Sample 1)
├── groundtruth_d65/
│   └── Sample1_D65.xlsx        # Groundtruth under D65 lighting
├── groundtruth_d75/
│   └── Sample1_D75.xlsx
├── groundtruth_A/
│   └── Sample1_A.xlsx
├── groundtruth_C/
│   └── Sample1_C.xlsx
├── README_data.txt            # Sample data description
🔎 Format Notes
input/1.xlsx: 49 rows × 13 columns (1 wavelength column + 12 spectral columns)
Groundtruth files: 41 × 49 excitation-emission matrices
See data/README_data.txt for full format details
⚠️ The full dataset (41 samples × 4 modes) is not included.
To request full data access, please contact: your_email@example.com

🧠 Model Overview
The main architecture consists of:

Input Embedding: Linear(6 → 128) for both incident and emission vectors
Interaction Mechanism: Element-wise multiplication
1D CNN Layers: Capture local spectral features
Transformer Encoder: Capture long-range dependencies
Decoder: Fully connected layers with Softplus to predict non-negative EEM
Supported Models:
ExcitationSpectrumModel_CNN
ExcitationSpectrumModel_CNN_Transformer
ExcitationSpectrumModel_CNN_Transformer_Improved
ExcitationSpectrumModel_CNN_UNet
ExcitationSpectrumModel_CNN_ResNet
ExcitationSpectrumModel_CNN_MultiScale
ExcitationSpectrumModel_CNN_Interaction
📊 Loss Functions
The total loss is a weighted combination of:

Mean Squared Error (MSE)
Generalized Feature Correlation (GFC Loss)
Smoothness Regularization
🔍 Loss function definitions are in: models/losses.py

🏋️‍♀️ Training
To train the model:

python train.py
During training, the following will be saved:

Best model (.pth)
Loss history (.xlsx)
Loss curves (.png)
You can modify train.py to switch models or use only Sample 1 for testing:


X = X[1:2]
Y = Y[1:2]
📈 Visualize Loss Curves

python plot_loss.py
Generates loss curve plots from training log.

📦 Requirements
Tested with:

Python ≥ 3.8
PyTorch ≥ 1.12
Dependencies:

torch>=1.12.1
pandas
numpy
scikit-learn
matplotlib
scipy
openpyxl
Install with:

pip install -r requirements.txt
📌 Notes
Input shape: [batch_size, 49, 13]
Output shape: [batch_size, 41, 49]
Output is non-negative due to Softplus in the decoder

