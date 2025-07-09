import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from Dataset import create_10class_dataloaders
from Model import FTUCNNMulticlass
import matplotlib.pyplot as plt
from Dataset import set_seed

# Set seed
set_seed(42)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset path (update as needed)
dataset_path = r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs\Dataset\Grouped Ftu test"

# Load data with new split
train_loader, val_loader, test_loader, class_names = create_10class_dataloaders(dataset_path, batch_size=32)

# Initialize model
model = FTUCNNMulticlass().to(device)

# Loss
loss_function = nn.CrossEntropyLoss()

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
    model_path = os.path.join(current_dir, "model_res18_2.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

# === Training function with validation ===
def train_model(epochs=20):
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
    print("🚀 Starting training...")
    start_time = time.time()

    train_model()

    end_time = time.time()
    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"⏱️ Training completed in {int(minutes)} min {int(seconds)} sec")