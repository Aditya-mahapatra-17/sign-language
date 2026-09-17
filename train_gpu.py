import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "asl_alphabet_train", "asl_alphabet_train")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_model(epochs=8, batch_size=64, lr=0.001, val_split=0.15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(f"🔥 TRAINING ASL MODEL ON: {device.type.upper()}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    print("==================================================")

    # 1. Transforms (No horizontal flip to preserve sign handedness)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset
    print(f"\n📂 Loading dataset from: {DATA_DIR}")
    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    classes = full_dataset.classes
    num_classes = len(classes)
    print(f"✅ Found {len(full_dataset)} images across {num_classes} classes: {classes}")

    # Save class names mapping
    class_names_path = os.path.join(MODELS_DIR, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(classes, f)
    print(f"📝 Saved class names to {class_names_path}")

    # Train / Val Split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Wrap subsets with respective transforms
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return self.transform(x), y

    train_data = TransformedSubset(train_subset, train_transform)
    val_data = TransformedSubset(val_subset, val_transform)

    num_workers = 2 if os.name == 'nt' else 4
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 3. Model: Pretrained MobileNetV2
    print("\n🏗️ Initializing MobileNetV2 (Pretrained on ImageNet)...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    
    # Replace final classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    model = model.to(device)

    # 4. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "sign_language_pytorch.pt")

    print(f"\n🚀 Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'class_names': classes,
                'val_acc': best_val_acc
            }, best_model_path)
            print(f"   ⭐ New best validation accuracy: {best_val_acc:.2f}% (Saved to {best_model_path})")

    print(f"\n🎉 Training complete! Best Model Accuracy: {best_val_acc:.2f}%")
    return best_model_path

if __name__ == "__main__":
    train_model(epochs=6, batch_size=64, lr=0.001)
