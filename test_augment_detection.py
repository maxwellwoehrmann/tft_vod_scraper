import os
import cv2
import torch
import json
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torchvision.transforms as transforms
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec

# Configuration variables - from image_divider.py
ROI_X = 1270
ROI_Y = 220
ROI_WIDTH = 160
ROI_HEIGHT = 160
CONF_THRESHOLD = 0.25
SUB_IMAGE_WIDTH = 30
SUB_IMAGE_HEIGHT = 30

def extract_roi(image_path):
    """Extract region of interest from a full-sized image"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None, None
    
    # Extract ROI
    roi = img[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
    
    # Check if ROI is valid
    if roi.shape[0] != ROI_HEIGHT or roi.shape[1] != ROI_WIDTH:
        print(f"Warning: Extracted ROI dimensions {roi.shape} don't match expected {ROI_WIDTH}x{ROI_HEIGHT}")
    
    return img, roi

def load_model(model_path, classes_path, model_type="resnet34"):
    """Load classification model - from test_augment_model.py"""
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
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create model with correct architecture
    if model_type == "resnet18":
        from torchvision import models
        model = models.resnet18(pretrained=False)
        # Modify first conv layer to match the saved model
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the maxpool layer for small input images
        model.maxpool = torch.nn.Identity()
    elif model_type == "resnet34":
        from torchvision import models
        model = models.resnet34(pretrained=False)
        # Apply same modifications
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
    
    # Determine number of classes from the model weights
    state_dict = torch.load(model_path, map_location='cpu')
    fc_weight_shape = state_dict['fc.weight'].shape
    num_classes = fc_weight_shape[0]
    print(f"Model's final layer has {num_classes} outputs")
    
    # Adjust final layer to match number of classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Load the saved state dict
    model.load_state_dict(state_dict)
    
    # Move model to device and set eval mode
    model = model.to(device)
    model.eval()
    
    return model, classes, device

def predict_augment(model, image, classes, device, top_k=5):
    """Predict the class of a single augment image"""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
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

def split_and_save_box(roi, box, base_filename, output_dir):
    """Split detected box into 30x30 sub-images and save them - from image_divider.py"""
    # Extract box coordinates
    x1, y1, x2, y2, conf, cls = box
    
    # Convert to integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Extract the box image
    box_img = roi[y1:y2, x1:x2]
    
    # Get dimensions
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Normalize size if slightly off
    expected_height = SUB_IMAGE_HEIGHT
    if abs(box_height - expected_height) <= 5:  # Allow small deviation
        # Resize to exact height if close
        box_img = cv2.resize(box_img, (box_width, expected_height))
        box_height = expected_height
    
    # Calculate how many sub-images we need
    num_splits = max(1, round(box_width / SUB_IMAGE_WIDTH))
    
    # If the box is much wider than 30px, we'll split it
    sub_images = []
    
    if num_splits == 1 or box_width <= SUB_IMAGE_WIDTH + 5:
        # Just resize to exactly 30x30 if it's a single box or close to it
        resized_img = cv2.resize(box_img, (SUB_IMAGE_WIDTH, SUB_IMAGE_HEIGHT))
        sub_images.append(resized_img)
    else:
        # Split into multiple boxes
        split_width = box_width / num_splits
        
        for i in range(num_splits):
            start_x = int(i * split_width)
            end_x = int((i + 1) * split_width)
            
            # Extract sub-image
            sub_img = box_img[:, start_x:end_x]
            
            # Resize to exactly 30x30
            resized_sub = cv2.resize(sub_img, (SUB_IMAGE_WIDTH, SUB_IMAGE_HEIGHT))
            sub_images.append(resized_sub)
    
    # Save all sub-images and return their paths
    sub_image_paths = []
    for i, sub_img in enumerate(sub_images):
        # Create filename: original_0001_box1_part1.jpg
        sub_filename = f"{base_filename}_box{int(conf*100):03d}_part{i+1}.jpg"
        output_path = os.path.join(output_dir, sub_filename)
        
        # Convert from BGR to RGB for saving with PIL
        sub_img_rgb = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
        Image.fromarray(sub_img_rgb).save(output_path)
        sub_image_paths.append(output_path)
    
    return sub_images, sub_image_paths

def create_results_visualization(augment_images, predictions, output_path, original_img=None):
    """Create a visual layout showing the detected augments and their predictions"""
    fig = plt.figure(figsize=(12, 8))
    
    # If we have the original image, show it at the top
    if original_img is not None:
        gs = GridSpec(2, 3, height_ratios=[3, 2], figure=fig)
        ax_orig = plt.subplot(gs[0, :])
        ax_orig.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        ax_orig.set_title("Original Image with ROI")
        ax_orig.axis('off')
        
        # Draw ROI rectangle
        import matplotlib.patches as patches
        rect = patches.Rectangle(
            (ROI_X, ROI_Y), ROI_WIDTH, ROI_HEIGHT, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax_orig.add_patch(rect)
    else:
        gs = GridSpec(1, 3, figure=fig)
    
    # Show each augment with its predictions
    num_augments = min(3, len(augment_images))
    for i in range(num_augments):
        # Create subplot for image
        ax_img = plt.subplot(gs[1 if original_img is not None else 0, i])
        ax_img.imshow(cv2.cvtColor(augment_images[i], cv2.COLOR_BGR2RGB))
        ax_img.set_title(f"Augment {i+1}")
        ax_img.axis('off')
        
        # Add predictions as text
        preds = predictions[i]
        pred_text = "\n".join([f"{p['class']}: {p['probability']:.2f}" for p in preds[:3]])
        ax_img.text(
            0.5, -0.15, pred_text, 
            ha='center', va='top', transform=ax_img.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def process_images(
    images_dir, 
    yolo_model_path,
    classifier_model_path,
    classes_path,
    classifier_model_type="resnet34",
    output_base_dir="results"
):
    """Process all images in a directory"""
    # Load YOLO model
    print(f"Loading YOLO model from {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    
    # Load classification model
    print(f"Loading classifier model from {classifier_model_path}")
    classifier, classes, device = load_model(
        classifier_model_path, 
        classes_path,
        model_type=classifier_model_type
    )
    
    # Get image files
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    print(f"Processing {len(image_files)} images")
    
    # Process each image
    for img_path in image_files:
        print(f"Processing {img_path}")
        base_filename = Path(img_path).stem
        
        # Create output directory
        output_dir = os.path.join(output_base_dir, base_filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract ROI
        full_img, roi = extract_roi(img_path)
        
        if roi is None:
            print(f"  Failed to extract ROI")
            continue
        
        # Save ROI for reference
        roi_path = os.path.join(output_dir, f"{base_filename}_roi.jpg")
        cv2.imwrite(roi_path, roi)
        
        # Save original image copy
        orig_path = os.path.join(output_dir, f"{base_filename}_original.jpg")
        cv2.imwrite(orig_path, full_img)
        
        # Run detection on ROI
        detection_results = yolo_model.predict(
            source=roi,
            conf=CONF_THRESHOLD,
            verbose=False
        )
        
        # Check if any boxes detected
        if len(detection_results) > 0 and len(detection_results[0].boxes) > 0:
            boxes = detection_results[0].boxes.data.cpu().numpy()
            print(f"  Detected {len(boxes)} boxes")
            
            # Process up to 3 augments
            augment_limit = min(3, len(boxes))
            boxes = boxes[:augment_limit]
            
            # Split boxes and get sub-images
            all_sub_images = []
            all_sub_paths = []
            
            for box in boxes:
                sub_images, sub_paths = split_and_save_box(
                    roi, box, base_filename, output_dir
                )
                all_sub_images.extend(sub_images)
                all_sub_paths.extend(sub_paths)
            
            # Get predictions for each sub-image
            all_predictions = []
            for sub_img in all_sub_images:
                preds = predict_augment(classifier, sub_img, classes, device, top_k=5)
                all_predictions.append(preds)
            
            # Create visualization
            if all_sub_images:
                vis_path = os.path.join(output_dir, f"{base_filename}_results.jpg")
                create_results_visualization(
                    all_sub_images[:3],  # Limit to 3 augments
                    all_predictions[:3],  # Corresponding predictions
                    vis_path,
                    original_img=full_img
                )
                print(f"  Created visualization: {vis_path}")
            else:
                print("  No valid sub-images extracted")
        else:
            print("  No boxes detected")

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "images_dir": "temp/frames",
        "yolo_model_path": "runs/box_detection/weights/best.pt",
        "classifier_model_path": "augment_models/best_model.pth",
        "classes_path": "training_dataset/augment_dataset/class_mapping.json",
        "classifier_model_type": "resnet34",
        "output_base_dir": "augment_results"
    }
    
    # Process all images
    process_images(**CONFIG)