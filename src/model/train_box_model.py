import os
import torch
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

def train_model(
    data_yaml="dataset/data.yaml",
    output_dir="runs",
    epochs=50,
    patience=5,
    batch_size=16,
    img_size=160,
    device=None
):
    """Train a YOLO nano model for box detection"""
    # Set device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS (Apple Silicon)")
        else:
            device = "cpu"
            print("Using CPU")
    
    # Create model directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = YOLO("yolov8n.yaml")  # Create a new nano model
    
    # Set training arguments
    training_args = {
        "data": data_yaml,
        "epochs": epochs,
        "patience": patience,
        "batch": batch_size,
        "imgsz": img_size,
        "device": device,
        "project": output_dir,
        "name": "box_detection",
        "exist_ok": True,
        
        # Learning rate settings for slower, stable progression
        "lr0": 0.0005,  # Initial learning rate (half default)
        "lrf": 0.01,     # Final LR as fraction of initial
        "cos_lr": True,  # Use cosine scheduler
        
        # Regularization to prevent overfitting
        "weight_decay": 0.0005,
        "dropout": 0.1,  # Add dropout to prevent overfitting
        
        # Warmup settings
        "warmup_epochs": 5.0,
        "warmup_momentum": 0.5,
        
        # Augmentation (light to medium)
        "hsv_h": 0.015,  # HSV-Hue augmentation
        "hsv_s": 0.3,    # HSV-Saturation augmentation
        "hsv_v": 0.2,    # HSV-Value augmentation
        "degrees": 0,    # Rotation (disabled)
        "translate": 0.1, # Translation
        "scale": 0.2,    # Scale
        "fliplr": 0.5,   # Horizontal flip
        "mosaic": 0.0,   # NOOOO Mosaic
        
        # Save best weights
        "save_period": -1,  # Save at the end
    }
    
    # Start training
    print(f"Starting training for {epochs} epochs...")
    results = model.train(**training_args)
    
    print("Training complete")
    return model, results

def validate_model(model, data_yaml="dataset/data.yaml"):
    """Validate the trained model"""
    # Load dataset info
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    # Run validation
    print("Validating model...")
    metrics = model.val(data=data_yaml)
    
    print("\nValidation Metrics:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics

def visualize_predictions(model, data_dir="training_dataset/box_dataset", num_samples=25):
    """Visualize model predictions on validation samples"""
    val_dir = os.path.join(data_dir, "images", "val")
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) 
                if f.endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
    
    # Make predictions and visualize
    print("\nGenerating visualizations...")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    for i, img_path in enumerate(val_files):
        results = model.predict(img_path, conf=0.25)
        result_img = results[0].plot()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(result_img)
        plt.axis('off')
        plt.title(f"Validation Sample {i+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"val_pred_{i+1}.png"))
        plt.close()
    
    print(f"Saved visualizations to {results_dir}/")

def export_model(model, output_path="augment_models/yolo_nano_box_detector.pt"):
    """Export the model for inference"""
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting model to {output_path}")
    model.export(format="torchscript", imgsz=160)
    
    # Copy the exported model to the specified path
    exported = Path("runs/box_detection/weights/best.torchscript")
    os.rename(exported, output_path.replace(".pt", ".torchscript"))
    
    print(f"Model exported successfully")

if __name__ == "__main__":
    # Training parameters
    params = {
        "data_yaml": "training_dataset/box_dataset/data.yaml",
        "output_dir": "runs",
        "epochs": 15,
        "patience": 5,
        "batch_size": 16,
        "img_size": 160,
        # "device" will be auto-detected
    }
    
    # Train the model
    model, results = train_model(**params)
    
    # Validate
    metrics = validate_model(model, params["data_yaml"])
    
    # Visualize results
    visualize_predictions(model)

    export_model(model)