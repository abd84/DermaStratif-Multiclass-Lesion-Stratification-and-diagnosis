import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torch import nn, optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set device (MPS for macOS, fallback to CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Paths to files and folders
images_folder = "/Users/abdullah/Downloads/archive/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
ground_truth_csv = "/Users/abdullah/Downloads/archive/ISIC_2019_Training_GroundTruth.csv"
metadata_csv = "/Users/abdullah/Downloads/archive/ISIC_2019_Training_Metadata.csv"

# Load the CSV files
ground_truth = pd.read_csv(ground_truth_csv)
metadata = pd.read_csv(metadata_csv)

# Merge the ground truth and metadata files on the 'image' column
merged_data = pd.merge(ground_truth, metadata, on='image')

# Handle missing data
merged_data['age_approx'] = merged_data['age_approx'].fillna(merged_data['age_approx'].median())
merged_data['sex'] = merged_data['sex'].fillna('Unknown')
merged_data['anatom_site_general'] = merged_data['anatom_site_general'].fillna('Unknown')
merged_data = merged_data.drop(columns=['lesion_id'])

# Create a single label column ('diagnosis') from one-hot encoded columns
diagnosis_columns = ground_truth.columns[1:]
merged_data['diagnosis'] = merged_data[diagnosis_columns].idxmax(axis=1)

# Add the full image path to the dataframe
merged_data['image_path'] = merged_data['image'].apply(lambda x: os.path.join(images_folder, f"{x}.jpg"))

# Map diagnosis labels to integers
class_indices = {label: idx for idx, label in enumerate(merged_data['diagnosis'].unique())}
merged_data['diagnosis'] = merged_data['diagnosis'].map(class_indices)

# Split the data into training and validation sets
train_data, val_data = train_test_split(
    merged_data,
    test_size=0.2,
    stratify=merged_data['diagnosis'],
    random_state=42
)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=list(class_indices.values()),
    y=train_data['diagnosis']
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Define the Dataset class
class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        label = row['diagnosis']

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
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

# if __name__ == "__main__":
#     # Load dataset, define transformations, and set up DataLoaders
#     train_dataset = SkinLesionDataset(train_data, transform=train_transforms)
#     val_dataset = SkinLesionDataset(val_data, transform=val_transforms)

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)  # Larger batch size
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

#     # Define model, optimizer, and loss
#     model = efficientnet_b0(weights="DEFAULT")  # Use weights instead of pretrained
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_indices))
#     model.to(device)

#     criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Learning rate scheduler
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

#     # Mixed precision training setup
#     use_mixed_precision = torch.cuda.is_available()  # Check if mixed precision is supported
#     scaler = torch.amp.GradScaler() if use_mixed_precision else None

#     # Early stopping parameters
#     patience = 3
#     best_val_loss = float('inf')
#     early_stop_counter = 0

#     # Training loop
#     num_epochs = 20
#     for epoch in range(num_epochs):
#         print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
#         model.train()
#         running_loss = 0.0
#         correct_train = 0
#         total_train = 0

#         # Training phase with progress bar
#         for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training")):
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()

#             if use_mixed_precision:
#                 with torch.amp.autocast():  # Use mixed precision if supported
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#             running_loss += loss.item()

#             # Calculate training accuracy
#             _, predicted = torch.max(outputs, 1)
#             total_train += labels.size(0)
#             correct_train += (predicted == labels).sum().item()

#             if (batch_idx + 1) % 10 == 0:
#                 print(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

#         # Calculate and display average training accuracy
#         train_acc = 100 * correct_train / total_train
#         print(f"Average Training Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {train_acc:.2f}%")

#         # Validation phase with progress bar
#         model.eval()
#         val_loss = 0.0
#         correct_val = 0
#         total_val = 0
#         with torch.no_grad():
#             for inputs, labels in tqdm(val_loader, desc="Validating"):
#                 inputs, labels = inputs.to(device), labels.to(device)

#                 if use_mixed_precision:
#                     with torch.amp.autocast():  # Use mixed precision if supported
#                         outputs = model(inputs)
#                         loss = criterion(outputs, labels)
#                 else:
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)

#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 total_val += labels.size(0)
#                 correct_val += (predicted == labels).sum().item()

#         # Calculate and display validation accuracy
#         val_acc = 100 * correct_val / total_val
#         print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_acc:.2f}%")

#         # Save the best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             early_stop_counter = 0
#             torch.save(model.state_dict(), 'best_model.pth')
#             print("Best model saved!")
#         else:
#             early_stop_counter += 1

#         # Check for early stopping
#         if early_stop_counter >= patience:
#             print("Early stopping triggered.")
#             break

#         # Update learning rate
#         scheduler.step(val_loss) 




