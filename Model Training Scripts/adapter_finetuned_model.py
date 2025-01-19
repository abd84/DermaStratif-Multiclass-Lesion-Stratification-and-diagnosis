import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torch import nn, optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

images_folder = "/Users/abdullah/Downloads/archive/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
ground_truth_csv = "/Users/abdullah/Downloads/archive/ISIC_2019_Training_GroundTruth.csv"
metadata_csv = "/Users/abdullah/Downloads/archive/ISIC_2019_Training_Metadata.csv"
ground_truth = pd.read_csv(ground_truth_csv)
metadata = pd.read_csv(metadata_csv)

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

# Define Adapter
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.down_project(x))
        x = self.up_project(x)
        return x + residual

# Define model with adapters
class EfficientNetWithAdapter(nn.Module):
    def __init__(self, base_model, adapter_dim=64, num_classes=8):
        super(EfficientNetWithAdapter, self).__init__()
        self.features = base_model.features  
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling to flatten spatial dimensions
        self.adapters = nn.ModuleList([Adapter(1280, adapter_dim)])  # Adapter layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, num_classes)  # Classifier for final predictions
        )

    def forward(self, x):
        x = self.features(x)  # Extract features
        x = self.pool(x)  # Apply global pooling (batch_size, 1280, 1, 1)
        x = x.flatten(start_dim=1)  # Flatten to (batch_size, 1280)
        for adapter in self.adapters:
            x = adapter(x)  # Pass through adapter layers
        x = self.classifier(x)  # Final classification
        return x


if __name__ == "__main__":
    train_dataset = SkinLesionDataset(train_data, transform=train_transforms)
    val_dataset = SkinLesionDataset(val_data, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Initialize the model
    base_model = efficientnet_b0(weights="DEFAULT")
    num_classes = len(class_indices)
    model = EfficientNetWithAdapter(base_model, adapter_dim=64, num_classes=num_classes).to(device)

    # Load the saved model weights safely
    saved_model_path = '/Users/abdullah/Desktop/VS/best_model1.pth'
    if os.path.exists(saved_model_path):
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint, strict=False)  # Allow partial loading
        print(f"Loaded saved model weights from {saved_model_path}, ignoring missing adapter keys.")
    else:
         print(f"No saved model found at {saved_model_path}. Starting training from scratch.")

    # Freeze the pre-trained layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Define optimizer and loss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Training loop
    num_epochs = 10
    patience = 3
    best_val_loss = float('inf')
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

        # Validation phase
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

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), saved_model_path)
            print("Best model with adapters saved!")
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

