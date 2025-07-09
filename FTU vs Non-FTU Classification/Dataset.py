import os
import torch
import pydicom
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random

# === SET SEED FOR REPRODUCIBILITY ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
    transforms.Normalize(mean=[0.5], std=[0.5])
])

nonftu_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# === CUSTOM DATASET CLASS ===
class DicomDataset(Dataset):
    def __init__(self, image_paths, labels, ftu_transform=None, nonftu_transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.ftu_transform = ftu_transform
        self.nonftu_transform = nonftu_transform

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

            if label == 1 and self.ftu_transform:
                img = self.ftu_transform(img)
            elif label == 0 and self.nonftu_transform:
                img = self.nonftu_transform(img)

            return img, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"❌ Skipped file: {os.path.basename(path)} — {e}")
            return self.__getitem__((index + 1) % len(self.image_paths))

# === LOAD IMAGE PATHS GROUPED BY CASE ID ===
def load_paths_by_case(ftu_dir, nonftu_dir):
    case_dict = {}
    for file in os.listdir(ftu_dir):
        if file.endswith(".dcm"):
            case = file.split("_")[1]
            case_dict.setdefault(case, []).append((os.path.join(ftu_dir, file), 1))
    for file in os.listdir(nonftu_dir):
        if file.endswith(".dcm"):
            case = file.split("_")[1]
            case_dict.setdefault(case, []).append((os.path.join(nonftu_dir, file), 0))
    return case_dict

# === CREATE DATALOADERS WITH 80/10/10 SPLIT ===
def create_dataloaders(ftu_dir, nonftu_dir, batch_size=64):
    set_seed(42)
    case_dict = load_paths_by_case(ftu_dir, nonftu_dir)
    cases = list(case_dict.keys())

    train_cases, temp_cases = train_test_split(cases, test_size=0.2, random_state=42)
    val_cases, test_cases = train_test_split(temp_cases, test_size=0.5, random_state=42)

    def flatten(cases_subset):
        paths, labels = [], []
        for case in cases_subset:
            for path, label in case_dict[case]:
                paths.append(path)
                labels.append(label)
        return paths, labels

    train_paths, train_labels = flatten(train_cases)
    val_paths, val_labels = flatten(val_cases)
    test_paths, test_labels = flatten(test_cases)

    train_dataset = DicomDataset(train_paths, train_labels, ftu_transform, nonftu_transform)
    val_dataset   = DicomDataset(val_paths, val_labels, ftu_transform, nonftu_transform)
    test_dataset  = DicomDataset(test_paths, test_labels, ftu_transform, nonftu_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  worker_init_fn=lambda id: np.random.seed(42 + id))
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, worker_init_fn=lambda id: np.random.seed(42 + id))
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, worker_init_fn=lambda id: np.random.seed(42 + id))

    print(f"✅ Split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    return train_loader, val_loader, test_loader

# === TESTING ===
if __name__ == "__main__":
    dataset_path = r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs\Dataset\FTU_NONFTU_Dataset"
    ftu_dir = os.path.join(dataset_path, "FTU")
    nonftu_dir = os.path.join(dataset_path, "NonFTU")
    train_loader, val_loader, test_loader = create_dataloaders(ftu_dir, nonftu_dir, batch_size=64)
