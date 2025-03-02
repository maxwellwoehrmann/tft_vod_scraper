import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import models
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shutil

def load_model(model_path, classes_path, model_type="resnet18"):
    # Load class names
    try:
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        
        # Convert string keys to integers if dictionary
        if isinstance(classes, dict):
            class_map = {}
            for key, value in classes.items():
                class_map[int(key)] = value
            classes = class_map
            
        print(f"Loaded class mapping with {len(classes)} classes")
    except Exception as e:
        print(f"Error loading classes from {classes_path}: {e}")
        print("Creating fallback class mapping")
        classes = {i: f"class_{i}" for i in range(1000)}  # Fallback with 1000 classes
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    
    # Create model with correct architecture
    if model_type == "resnet18":
        model = models.resnet18(pretrained=False)
        # Modify first conv layer to match the saved model
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the maxpool layer for small input images
        model.maxpool = nn.Identity()
    elif model_type == "resnet34":
        model = models.resnet34(pretrained=False)
        # Apply same modifications
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    
    # Determine number of classes from the model weights
    state_dict = torch.load(model_path, map_location='cpu')
    fc_weight_shape = state_dict['fc.weight'].shape
    num_classes = fc_weight_shape[0]
    print(f"Model's final layer has {num_classes} outputs")
    
    # Adjust final layer to match number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load the saved state dict
    model.load_state_dict(state_dict)
    
    # Move model to device BEFORE setting eval mode
    model = model.to(device)
    model.eval()
    
    return model, classes, device

def predict_augment(model, image_path, classes, device, top_k=5):
    """Predict the class of a single augment image"""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Move to same device as model
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    # Get top-k predictions
    probs, indices = torch.topk(probabilities, top_k)
    probs = probs.squeeze().cpu().numpy()
    indices = indices.squeeze().cpu().numpy()
    
    # Map indices to class names
    predictions = []
    for i, idx in enumerate(indices):
        # Convert numpy int to python int if necessary
        if isinstance(idx, np.integer):
            idx = idx.item()
            
        # Get class name safely
        if isinstance(classes, dict):
            class_name = classes.get(idx, f"unknown_class_{idx}")
        else:
            class_name = classes[idx] if 0 <= idx < len(classes) else f"unknown_class_{idx}"
            
        predictions.append({
            'class': class_name,
            'probability': float(probs[i])
        })
    
    return predictions

def test_augments(test_dir, model_path, classes_path, output_dir='augment_test_results', 
                 model_type='resnet34', top_k=5, min_confidence=0.0):
    """Test the model on a directory of augment images"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and classes
    model, classes, device = load_model(model_path, classes_path, model_type)
    
    # Find all images in test directory
    image_paths = []
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        image_paths.extend(list(Path(test_dir).glob(f'*{ext}')))
    
    if not image_paths:
        print(f"No images found in {test_dir}")
        return
    
    print(f"Found {len(image_paths)} images to test")
    
    # Process each image
    results = {}
    for img_path in tqdm(image_paths):
        predictions = predict_augment(model, img_path, classes, device, top_k)
        results[str(img_path)] = predictions
    
    # Save detailed results
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization of results
    create_results_visualization(image_paths, results, output_dir, top_k, min_confidence)
    
    # Generate summary statistics
    top1_correct = 0
    for img_path in image_paths:
        img_name = img_path.stem
        top_prediction = results[str(img_path)][0]['class']
        
        # If image filename contains the true class, check if it matches top prediction
        if img_name.startswith(top_prediction) or top_prediction in img_name:
            top1_correct += 1
    
    accuracy = top1_correct / len(image_paths) if image_paths else 0
    print(f"\nResults Summary:")
    print(f"Total images: {len(image_paths)}")
    print(f"Estimated accuracy: {accuracy:.2%}")
    print(f"Detailed results saved to {output_dir}")

def create_results_visualization(image_paths, results, output_dir, top_k=5, min_confidence=0.0):
    """Create a visual grid of test images with their predictions"""
    # Filter images based on minimum confidence
    filtered_paths = []
    for img_path in image_paths:
        if results[str(img_path)][0]['probability'] >= min_confidence:
            filtered_paths.append(img_path)
    
    if not filtered_paths:
        print("No images meet the minimum confidence threshold")
        return
    
    # Determine grid size
    n_images = len(filtered_paths)
    grid_size = min(5, n_images)  # Max 5 images per row
    n_rows = (n_images + grid_size - 1) // grid_size
    
    # Create visualizations in batches to avoid memory issues
    batch_size = min(25, n_images)  # 5x5 grid max per page
    for batch_start in range(0, n_images, batch_size):
        batch_end = min(batch_start + batch_size, n_images)
        batch_paths = filtered_paths[batch_start:batch_end]
        
        fig = plt.figure(figsize=(15, 3 * ((batch_end - batch_start + grid_size - 1) // grid_size)))
        gs = gridspec.GridSpec((batch_end - batch_start + grid_size - 1) // grid_size, grid_size)
        
        for i, img_path in enumerate(batch_paths):
            # Create subplot
            ax = plt.subplot(gs[i])
            
            # Display image
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            ax.axis('off')
            
            # Get predictions
            preds = results[str(img_path)]
            
            # Format title with top predictions
            title = f"Top predictions:\n"
            for j, pred in enumerate(preds[:min(3, top_k)]):
                title += f"{j+1}. {pred['class']}: {pred['probability']:.2%}\n"
            
            ax.set_title(title, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'results_grid_{batch_start//batch_size+1}.png'), dpi=150)
        plt.close(fig)
    
    # Create histogram of top-1 confidence
    plt.figure(figsize=(10, 6))
    confidences = [results[str(img_path)][0]['probability'] for img_path in image_paths]
    plt.hist(confidences, bins=20, alpha=0.7)
    plt.title('Distribution of Top-1 Confidence Scores')
    plt.xlabel('Confidence')
    plt.ylabel('Number of Images')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'confidence_histogram.png'))
    plt.close()

def extract_augments_from_strip(strip_image_path, output_dir, strip_width=90, augment_size=30):
    """Extract individual augments from a strip image"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load strip image
    strip_img = Image.open(strip_image_path).convert('RGB')
    
    # Calculate number of augments
    num_augments = strip_width // augment_size
    
    # Extract each augment
    for i in range(num_augments):
        x_start = i * augment_size
        augment = strip_img.crop((x_start, 0, x_start + augment_size, augment_size))
        
        # Save individual augment
        output_path = os.path.join(output_dir, f'augment_{i+1}_{Path(strip_image_path).stem}.png')
        augment.save(output_path)
    
    print(f"Extracted {num_augments} augments from {strip_image_path}")
    return [os.path.join(output_dir, f'augment_{i+1}_{Path(strip_image_path).stem}.png') for i in range(num_augments)]

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "test_dir": "split_boxes",  # Directory containing 30x30 augment images to test
        "model_path": "augment_models/best_model.pth",  # Path to trained model weights
        "classes_path": "training_dataset/augment_dataset/class_mapping.json",  # Path to class mapping file
        "output_dir": "augment_test_results",  # Directory to save results
        "model_type": "resnet34",  # Model architecture type
        "top_k": 5,  # Number of top predictions to show
        "min_confidence": 0.0  # Minimum confidence threshold for visualization
    }
    
    # Test the model
    test_augments(**CONFIG)