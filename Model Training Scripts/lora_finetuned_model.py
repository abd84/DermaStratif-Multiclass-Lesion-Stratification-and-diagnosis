
import os
import torch
import math
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Set device (MPS for macOS, fallback to CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Paths to files and folders
images_folder = "/Users/abdullah/Downloads/archive/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
ground_truth_csv = "/Users/abdullah/Downloads/archive/ISIC_2019_Training_GroundTruth.csv"
metadata_csv = "/Users/abdullah/Downloads/archive/ISIC_2019_Training_Metadata.csv"

# Load CSV files
ground_truth = pd.read_csv(ground_truth_csv)
metadata = pd.read_csv(metadata_csv)

# Merge and preprocess data
merged_data = pd.merge(ground_truth, metadata, on="image")
merged_data['age_approx'] = merged_data['age_approx'].fillna(merged_data['age_approx'].median())
merged_data['sex'] = merged_data['sex'].fillna("Unknown")
merged_data['anatom_site_general'] = merged_data['anatom_site_general'].fillna("Unknown")
merged_data = merged_data.drop(columns=['lesion_id'])
diagnosis_columns = ground_truth.columns[1:]
merged_data['diagnosis'] = merged_data[diagnosis_columns].idxmax(axis=1)
merged_data['image_path'] = merged_data['image'].apply(lambda x: os.path.join(images_folder, f"{x}.jpg"))
class_indices = {label: idx for idx, label in enumerate(merged_data['diagnosis'].unique())}
merged_data['diagnosis'] = merged_data['diagnosis'].map(class_indices)

# Train-validation split
train_data, val_data = train_test_split(
    merged_data, test_size=0.2, stratify=merged_data['diagnosis'], random_state=42
)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=list(class_indices.values()),
    y=train_data['diagnosis']
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Dataset class
class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        label = row['diagnosis']
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transforms
IMG_HEIGHT, IMG_WIDTH = 224, 224
train_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Datasets and DataLoaders
train_dataset = SkinLesionDataset(train_data, transform=train_transforms)
val_dataset = SkinLesionDataset(val_data, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# Define LoRA
class LoRA(nn.Module):
    def __init__(self, in_features, r=8, alpha=32):
        super(LoRA, self).__init__()
        self.down_proj = nn.Linear(in_features, r, bias=False)
        self.up_proj = nn.Linear(r, in_features, bias=False)
        self.scaling = alpha / r

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        return self.up_proj(self.down_proj(x)) * self.scaling

# Modify EfficientNet to include LoRA
class EfficientNetWithLoRA(nn.Module):
    def __init__(self, base_model, num_classes, r=8, alpha=32):
        super(EfficientNetWithLoRA, self).__init__()
        self.features = base_model.features  # Use pre-trained EfficientNet features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            LoRA(1280, r=r, alpha=alpha),  # LoRA applied here
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

# Load EfficientNet with LoRA
base_model = efficientnet_b0(weights="DEFAULT")
num_classes = len(class_indices)
model = EfficientNetWithLoRA(base_model, num_classes=num_classes, r=8, alpha=32).to(device)

# Load saved model (if exists)
saved_model_path = "/Users/abdullah/Desktop/VS/best_model1.pth"
# Load the saved model weights safely
if os.path.exists(saved_model_path):
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint, strict=False)  # Allow partial loading
    print(f"Loaded saved model weights from {saved_model_path}, ignoring mismatched keys.")
else:
    print(f"No saved model found at {saved_model_path}. Training from scratch.")


# Optimizer and Loss Function
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Training loop
num_epochs = 10
patience = 3
best_val_loss = float("inf")
early_stop_counter = 0

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_acc = 100 * correct_train / total_train
    print(f"Training Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {train_acc:.2f}%")

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_acc = 100 * correct_val / total_val
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_acc:.2f}%")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), saved_model_path)
        print("Best model with LoRA saved!")
    else:
        early_stop_counter += 1

    # Early stopping
    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break
