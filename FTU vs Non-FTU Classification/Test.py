import os
import torch
import torch.nn as nn
import time
from Dataset import create_dataloaders
from Model import FTUCNN
from Dataset import set_seed

# Set seed
set_seed(42)

# Set dataset path
dataset_path = r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs\Dataset\FTU_NONFTU_Dataset"
ftu_dir = os.path.join(dataset_path, "FTU")
nonftu_dir = os.path.join(dataset_path, "NonFTU")

# Load test data
_, _, test_loader = create_dataloaders(ftu_dir, nonftu_dir, batch_size=64)

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
model = FTUCNN().to(device)  # Ensure model is initialized correctly
model.load_state_dict(torch.load(r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs\FTU vs Non-FTU Classification\model_FTU_nonftu_1.pth"))

#Loss function
loss_function = nn.CrossEntropyLoss()

# Validation
def test_model():
    """Function to evaluate the model on test data"""
    start = time.time()
    total = 0
    correct = 0
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # No gradients needed during testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get predicted class

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    avg_test_loss = running_loss/len(test_loader)
    end = time.time()
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f'Execution Time: {end - start:.2f} seconds')

    return all_labels, all_predictions, avg_test_loss

# Run the test function
if __name__ == "__main__":
    all_labels, all_predictions, test_loss = test_model()

