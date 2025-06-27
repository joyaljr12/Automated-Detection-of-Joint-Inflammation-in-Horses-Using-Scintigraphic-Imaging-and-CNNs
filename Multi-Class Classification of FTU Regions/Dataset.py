import os
import torch
import pydicom
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random

# === SET SEED ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === IMAGE TRANSFORMATIONS ===
ftu_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.ColorJitter(contrast=0.1),
    transforms.RandomRotation(10),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# === CUSTOM DATASET CLASS ===
class FTU10ClassDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        label = self.labels[index]
        try:
            dcm = pydicom.dcmread(path, force=True)
            if not hasattr(dcm, 'pixel_array'):
                raise RuntimeError("No pixel data")
            arr = dcm.pixel_array.astype(np.float32)

            if arr.ndim == 2:
                ptp = np.ptp(arr)
                arr = (255 * (arr - arr.min()) / ptp).astype(np.uint8) if ptp != 0 else np.zeros_like(arr)
                img = Image.fromarray(arr).convert("L")
            elif arr.ndim == 3 and arr.shape[0] > 1:
                slice_2d = arr[arr.shape[0] // 2]
                ptp = np.ptp(slice_2d)
                slice_2d = (255 * (slice_2d - slice_2d.min()) / ptp).astype(np.uint8) if ptp != 0 else np.zeros_like(slice_2d)
                img = Image.fromarray(slice_2d).convert("L")
            else:
                raise RuntimeError(f"Unsupported shape: {arr.shape}")

            img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"❌ Skipped file: {os.path.basename(path)} — {e}")
            self.skipped_files.append(path)
            return self.__getitem__((index + 1) % len(self.image_paths))

# === FUNCTION TO CREATE DATALOADERS ===
def create_10class_dataloaders(ftu_grouped_dir, batch_size=64):
    set_seed(42)
    class_names = sorted(os.listdir(ftu_grouped_dir))
    class_to_label = {cls: i for i, cls in enumerate(class_names)}

    image_paths = []
    labels = []

    for cls in class_names:
        class_dir = os.path.join(ftu_grouped_dir, cls)
        for fname in os.listdir(class_dir):
            if fname.endswith(".dcm"):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_label[cls])

    # Stratified split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

    train_set = FTU10ClassDataset(train_paths, train_labels, ftu_transform)
    val_set   = FTU10ClassDataset(val_paths, val_labels, ftu_transform)
    test_set  = FTU10ClassDataset(test_paths, test_labels, ftu_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

   

    print(f"✅ Dataset loaded: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")
    return train_loader, val_loader, test_loader, class_names

# === TEST RUN ===
if __name__ == "__main__":
    ftu_dir = r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs testing\Labelled image Classification\Grouped Ftu Dataset"
    create_10class_dataloaders(ftu_dir)
