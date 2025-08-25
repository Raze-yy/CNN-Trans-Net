# train.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models.cnn_models import ExcitationSpectrumModel_CNN_Transformer
from models.losses import ExcitationSpectrumLoss
from utils.data_loader import load_combined_spectra, load_groundtruth
from utils.preprocessing import process_input_spectra

# Load and preprocess data
input_raw = load_combined_spectra("data/input")
groundtruth = load_groundtruth("data/groundtruth_d65")

input_spectra = process_input_spectra(input_raw)

# Convert to tensors
X = torch.tensor(input_spectra, dtype=torch.float32)
Y = torch.tensor(groundtruth, dtype=torch.float32)

# Train/val split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=8, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=8)

# Model, loss, optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ExcitationSpectrumModel_CNN_Transformer().to(device)
criterion = ExcitationSpectrumLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(1000):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: train loss = {total_loss / len(train_loader):.6f}")