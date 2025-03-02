import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import shutil
from sklearn.metrics import f1_score

class AugmentDataset(Dataset):
    """Dataset for augment classification training"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset
            split (str): 'train' or 'val'
            transform (callable, optional): Transform to apply to images
        """
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split
        self.transform = transform
        
        # Get class names from directory structure
        self.classes = [d.name for d in self.split_dir.iterdir() if d.is_dir()]
        self.classes.sort()  # Sort for reproducibility
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get all image files and their labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_loaders(data_dir, batch_size=32):
    """Create training and validation data loaders"""
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=3, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = AugmentDataset(data_dir, split='train', transform=train_transform)
    val_dataset = AugmentDataset(data_dir, split='val', transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes

def create_model(num_classes, model_type="resnet34", pretrained=True):
    """Create a model for augment classification with adaptations for small images"""
    print(f"Creating {model_type} model for {num_classes} classes...")
    
    if model_type == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        # Adapt for small images (30x30)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove maxpool layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_type == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        # Adapt for small images (30x30)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove maxpool layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        # Adapt for small images (30x30)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove maxpool layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_type == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        # Efficientnet doesn't need as many adaptations for small images
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def compute_metrics(outputs, labels):
    """Compute various classification metrics"""
    # Regular accuracy (top-1)
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    
    # Top-5 accuracy (if applicable - classes must be >= 5)
    _, top5_pred = outputs.topk(min(5, outputs.size(1)), 1, True, True)
    top5_correct = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
    top5_accuracy = top5_correct / total
    
    # F1 score (macro-averaged)
    f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'f1_score': f1
    }

def train_model(model, train_loader, val_loader, classes, device, num_epochs=50, patience=15, 
                accumulation_steps=2, use_mixed_precision=True):
    """Train the augment classification model with advanced techniques"""
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Setup for mixed precision training if available
    scaler = None
    if use_mixed_precision:
        # Check if mixed precision is supported
        if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            print("Using mixed precision training")
        else:
            print("Mixed precision training not supported on this device")
    
    # Training metrics
    train_losses = []
    val_losses = []
    train_metrics = {
        'accuracy': [], 'top5_accuracy': [], 'f1_score': []
    }
    val_metrics = {
        'accuracy': [], 'top5_accuracy': [], 'f1_score': []
    }
    best_val_metric = 0.0  # Track best validation metric (F1 score)
    epochs_without_improvement = 0
    
    # Create output directory
    output_dir = Path("augment_models")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Using gradient accumulation with {accumulation_steps} steps")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        
        # Zero the gradients at the beginning
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass with mixed precision if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                
                # Step with gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard forward and backward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Normalize loss for gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                # Step with gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Update metrics
            running_loss += loss.item() * images.size(0) * (1 if scaler is None else accumulation_steps)
            all_outputs.append(outputs.detach())
            all_labels.append(labels)
            
            # Calculate batch metrics for progress bar
            batch_metrics = compute_metrics(outputs.detach(), labels)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() * (1 if scaler is None else accumulation_steps),
                'acc': f"{batch_metrics['accuracy']:.4f}",
                'f1': f"{batch_metrics['f1_score']:.4f}"
            })
        
        # Compute epoch-level metrics
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Combine all outputs and labels for complete metrics
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        epoch_metrics = compute_metrics(all_outputs, all_labels)
        
        # Store metrics
        for key, value in epoch_metrics.items():
            train_metrics[key].append(value)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        class_correct = [0] * len(classes)
        class_total = [0] * len(classes)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Update metrics
                running_loss += loss.item() * images.size(0)
                all_outputs.append(outputs)
                all_labels.append(labels)
                
                # Per-class accuracy
                _, predicted = outputs.max(1)
                c = predicted.eq(labels).cpu()
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                
                # Calculate batch metrics for progress bar
                batch_metrics = compute_metrics(outputs, labels)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': f"{batch_metrics['accuracy']:.4f}",
                    'f1': f"{batch_metrics['f1_score']:.4f}"
                })
        
        # Compute epoch-level validation metrics
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Combine all validation outputs and labels
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        epoch_metrics = compute_metrics(all_outputs, all_labels)
        
        # Store validation metrics
        for key, value in epoch_metrics.items():
            val_metrics[key].append(value)
        
        # Update learning rate based on F1 score
        scheduler.step(epoch_metrics['f1_score'])
        
        # Print epoch summary with all metrics
        print(f"\nEpoch {epoch+1}/{num_epochs} summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy'][-1]:.4f}, "
              f"Top-5: {train_metrics['top5_accuracy'][-1]:.4f}, F1: {train_metrics['f1_score'][-1]:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy'][-1]:.4f}, "
              f"Top-5: {val_metrics['top5_accuracy'][-1]:.4f}, F1: {val_metrics['f1_score'][-1]:.4f}")
        
        # Save model if F1 score improved
        current_metric = epoch_metrics['f1_score']  # Use F1 as the primary metric
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"Saved new best model with F1 score: {current_metric:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs without improvement")
            break
    
    # Print per-class statistics for best model
    print("\nLoading best model for per-class evaluation...")
    model.load_state_dict(torch.load(output_dir / "augment_best_model.pth"))
    model.eval()
    
    # Evaluate per-class performance
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            c = predicted.eq(labels).cpu()
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Save per-class accuracy
    class_accuracy = {}
    for i, cls in enumerate(classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            class_accuracy[cls] = float(acc)
            print(f"{cls}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            class_accuracy[cls] = 0.0
            print(f"{cls}: No samples")
    
    # Save class accuracy to file
    with open(output_dir / "class_accuracy.json", "w") as f:
        json.dump(class_accuracy, f, indent=2)
    
    # Save the final model
    torch.save(model.state_dict(), output_dir / "augment_final_model.pth")
    
    # Save class mapping
    with open(output_dir / "classes.json", "w") as f:
        json.dump(classes, f)
    
    # Save all metrics
    all_metrics = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }
    
    with open(output_dir / "training_metrics.json", "w") as f:
        # Convert numpy values to Python types for JSON serialization
        json_metrics = {k: [float(x) for x in v] if isinstance(v, list) else 
                        {k2: [float(x) for x in v2] for k2, v2 in v.items()}
                        for k, v in all_metrics.items()}
        json.dump(json_metrics, f, indent=2)
    
    # Plot all metrics
    fig = plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_metrics['accuracy'], label='Train')
    plt.plot(val_metrics['accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Top-5 Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(train_metrics['top5_accuracy'], label='Train')
    plt.plot(val_metrics['top5_accuracy'], label='Validation')
    plt.title('Top-5 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Top-5 Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 Score
    plt.subplot(2, 2, 4)
    plt.plot(train_metrics['f1_score'], label='Train')
    plt.plot(val_metrics['f1_score'], label='Validation')
    plt.title('F1 Score (Macro)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png")
    
    print(f"\nTraining completed.")
    print(f"Best validation F1 Score: {best_val_metric:.4f}")
    print(f"Models and plots saved to {output_dir}")
    
    return model, all_metrics

def export_model(model, classes, output_dir="augment_models"):
    """Export the model for inference"""
    # Save torchscript model
    output_dir = Path(output_dir)
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, 30, 30)
    if torch.backends.mps.is_available():
        example_input = example_input.to("mps")
        model = model.to("mps")
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(output_dir / "augment_classifier.pt")
    
    # Save class info
    with open(output_dir / "classes.json", "w") as f:
        json.dump(classes, f)
    
    print(f"Exported model to {output_dir}/augment_classifier.pt")

if __name__ == "__main__":
    # Configuration
    config = {
        "data_dir": "training_dataset/augment_dataset",
        "batch_size": 32,
        "num_epochs": 10,
        "patience": 5,
        "model_type": "resnet34", 
        "pretrained": True,
        "accumulation_steps": 2,  # Accumulate gradients over 2 batches
        "use_mixed_precision": True  # Use mixed precision if available
    }
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create data loaders
    train_loader, val_loader, classes = create_data_loaders(
        config["data_dir"], 
        batch_size=config["batch_size"]
    )
    
    print(f"Classes: {len(classes)} unique augment types")
    print(f"Training with {len(train_loader.dataset)} images, validating with {len(val_loader.dataset)} images")
    
    # Create model
    model = create_model(
        num_classes=len(classes), 
        model_type=config["model_type"], 
        pretrained=config["pretrained"]
    )
    model = model.to(device)
    
    # Print model summary
    print(f"Model: {config['model_type']} (adapted for small 30x30 images)")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    model, metrics = train_model(
        model, 
        train_loader, 
        val_loader, 
        classes,
        device,
        num_epochs=config["num_epochs"],
        patience=config["patience"],
        accumulation_steps=config["accumulation_steps"],
        use_mixed_precision=config["use_mixed_precision"]
    )
    
    # Export model
    export_model(model, classes)
    
    print("Done!")