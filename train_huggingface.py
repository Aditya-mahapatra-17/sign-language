import os
import time
import argparse
import random
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_cosine_schedule_with_warmup
from tqdm import tqdm
from PIL import Image

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "asl_alphabet_train", "asl_alphabet_train")
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "models", "hf_finetuned_siglip")

class FastASLDataset(Dataset):
    """
    Lightweight, memory-safe dataset for high-speed training on both WSL and Native OS.
    """str
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                pixel_values = self.transform(img)
            else:
                pixel_values = transforms.ToTensor()(img)
        return pixel_values, label

def fine_tune_hf_model(
    hf_base_model="prithivMLmods/Alphabet-Sign-Language-Detection",
    epochs=5,
    batch_size=32, # Safe batch size for Laptop GPUs & WSL memory limits
    learning_rate=5e-5,
    val_split=0.15,
    samples_per_class=500, # 500 per class = ~14,500 total images (Fast, memory-safe & high accuracy)
    freeze_backbone=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================================")
    print("🔥 MEMORY-SAFE FAST FINE-TUNING (SigLIP2 Vision Transformer)")
    print(f"   Base Model: {hf_base_model}")
    print(f"   Compute Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    print("==================================================================")

    # 1. Load HuggingFace Image Processor
    print("\n📦 Loading HuggingFace Processor...")
    processor = AutoImageProcessor.from_pretrained(hf_base_model)

    # 2. Prepare Dataset from data/raw/
    print(f"📂 Loading ASL dataset from: {DATA_DIR}")
    raw_dataset = datasets.ImageFolder(root=DATA_DIR)
    classes = raw_dataset.classes
    num_classes = len(classes)
    
    id2label = {i: c for i, c in enumerate(classes)}
    label2id = {c: i for i, c in enumerate(classes)}
    print(f"✅ Found {len(raw_dataset)} total images across {num_classes} classes.")

    # Filter balanced samples per class to prevent memory exhaustion
    if samples_per_class is not None and samples_per_class > 0:
        print(f"⚡ Selecting {samples_per_class} balanced images per class (Total: ~{samples_per_class * num_classes} images)...")
        class_samples = defaultdict(list)
        for path, label in raw_dataset.samples:
            class_samples[label].append((path, label))

        selected_samples = []
        random.seed(42)
        for label, items in class_samples.items():
            k = min(samples_per_class, len(items))
            selected_samples.extend(random.sample(items, k))
        random.shuffle(selected_samples)
        dataset_pool = selected_samples
        print(f"🎯 Filtered training pool: {len(dataset_pool)} images.")
    else:
        dataset_pool = raw_dataset.samples

    # Train / Validation split
    val_size = int(len(dataset_pool) * val_split)
    train_size = len(dataset_pool) - val_size
    train_samples = dataset_pool[:train_size]
    val_samples = dataset_pool[train_size:]

    # Data Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = FastASLDataset(train_samples, transform=train_transform)
    val_data = FastASLDataset(val_samples, transform=val_transform)

    # Use num_workers=0 to prevent WSL2/Windows memory duplication errors
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 3. Load Model with customized head
    print("\n🏗️ Initializing Model with customized classification head (29 classes)...")
    model = AutoModelForImageClassification.from_pretrained(
        hf_base_model,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    if freeze_backbone:
        print("🔒 Freezing Vision Transformer backbone (training classifier head only for fast speed & low memory)...")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    model = model.to(device)

    # 4. Optimizer, Scheduler & Scaler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_acc = 0.0
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    # 5. Training Loop
    print(f"\n🚀 Starting fine-tuning for {epochs} epochs ({len(train_loader)} batches per epoch)...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", unit="batch")
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values=pixel_values)
                    loss = criterion(outputs.logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()
            
            train_loss += loss.item() * pixel_values.size(0)
            preds = outputs.logits.argmax(dim=-1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

            current_acc = (train_correct / train_total) * 100
            current_loss = train_loss / train_total
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.1f}%"})

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        # Validation Loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]", unit="batch", leave=False)
        with torch.no_grad():
            for pixel_values, labels in val_pbar:
                pixel_values = pixel_values.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(pixel_values=pixel_values)
                        loss = criterion(outputs.logits, labels)
                else:
                    outputs = model(pixel_values=pixel_values)
                    loss = criterion(outputs.logits, labels)

                val_loss += loss.item() * pixel_values.size(0)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        epoch_time = time.time() - start_time

        print(f"\n📊 Epoch [{epoch:02d}/{epochs:02d}] Summary ({epoch_time:.1f}s): "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f"   ⭐ New best validation accuracy: {best_val_acc:.2f}%! Saving to {OUTPUT_MODEL_DIR}...")
            model.save_pretrained(OUTPUT_MODEL_DIR)
            processor.save_pretrained(OUTPUT_MODEL_DIR)
            print("   💾 Checkpoint saved successfully.\n")

    print(f"\n🎉 Fine-Tuning Complete! Best Model Accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model files successfully saved in: {OUTPUT_MODEL_DIR}")
    return OUTPUT_MODEL_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune HuggingFace SigLIP2 Model on ASL Dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to fine-tune")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--samples-per-class", type=int, default=500, help="Number of images per class (e.g. 500 = ~14.5k images total)")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Unfreeze vision transformer backbone for full fine-tuning")
    args = parser.parse_args()

    fine_tune_hf_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        samples_per_class=args.samples_per_class,
        freeze_backbone=not args.unfreeze_backbone
    )
