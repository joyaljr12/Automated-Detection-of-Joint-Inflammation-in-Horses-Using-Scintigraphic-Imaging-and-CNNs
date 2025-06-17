import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from Dataset import create_dataloaders
from Model import FTUCNN
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set dataset path
dataset_path = r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs testing\FTU & Non FTU classification\FTU_NONFTU_Dataset"
ftu_dir = os.path.join(dataset_path, "FTU")
nonftu_dir = os.path.join(dataset_path, "NonFTU")

# Load data with new split
train_loader, val_loader, _ = create_dataloaders(ftu_dir, nonftu_dir, batch_size=64)

# Initialize model
model = FTUCNN().to(device)

# Class weighting
all_labels = []
for _, labels in train_loader:
    all_labels.extend(labels.tolist())
num_ftu = sum(1 for l in all_labels if l == 1)
num_nonftu = sum(1 for l in all_labels if l == 0)
class_weights = torch.tensor([1.0 * num_ftu / num_nonftu, 1.0]).to(device)
loss_function = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

# === Compute validation loss ===
def compute_validation_loss(loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
    model.train()
    return val_loss / len(loader)

# === Validation Accuracy ===
def evaluate_accuracy(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return 100 * correct / total

# === Save model ===
def save_model(model):
    current_dir = os.path.dirname(os.path.abspath(__file__))  
    model_path = os.path.join(current_dir, "model_FTU_nonftu_1.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

# === Training function with validation ===
def train_model(epochs=10):
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        val_loss = compute_validation_loss(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_acc = evaluate_accuracy(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
              f"Train Acc: {100*correct/total:.2f}% - Val Acc: {val_acc:.2f}%")

    # Plot losses
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', marker='s')
    plt.title("Training & Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    save_model(model)

# === Main ===
if __name__ == "__main__":
    train_model()
