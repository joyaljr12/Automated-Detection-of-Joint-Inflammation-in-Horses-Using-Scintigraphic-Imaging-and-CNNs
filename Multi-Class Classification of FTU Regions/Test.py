import os
import torch
import torch.nn as nn
from Dataset import create_10class_dataloaders
from Model import FTUCNNMulticlass

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Set dataset path ===
dataset_path = r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs testing\Labelled image Classification\Grouped Ftu Dataset"

# === Load data ===
_, _, test_loader, class_names = create_10class_dataloaders(dataset_path, batch_size=32)

# === Initialize and load model ===
model = FTUCNNMulticlass().to(device)
model.load_state_dict(torch.load(r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs testing\Labelled image Classification\Training Model\model_res18.pth", map_location=device))
model.eval()

# === Loss function ===
loss_function = nn.CrossEntropyLoss()

# === Test function ===
def test_model():
    total = 0
    correct = 0
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    avg_test_loss = running_loss / len(test_loader)

    print(f"✅ Test Accuracy: {accuracy:.2f}%")
    print(f"✅ Average Test Loss: {avg_test_loss:.4f}")
    return all_labels, all_predictions, avg_test_loss, class_names

# === Run if script is executed ===
if __name__ == '__main__':
    test_model()
