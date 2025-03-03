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
from ..utils import string_match

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
        model = models.resnet18(weights=None)
        # Modify first conv layer to match the saved model
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the maxpool layer for small input images
        model.maxpool = torch.nn.Identity()
    elif model_type == "resnet34":
        from torchvision import models
        model = models.resnet34(weights=None)
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

def split_box(roi, box, base_filename):
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
    
    return sub_images

def process_images(player_frames, augments):
    yolo_model_path = "runs/box_detection/weights/best.pt"
    classifier_model_path = "augment_models/best_model.pth"
    classes_path = "training_dataset/augment_dataset/class_mapping.json"
    classifier_model_type = "resnet34"


    augments_dict = dict()

    #parse names from assets folder
    def load_augments(color):
        return [path.stem.lower() for path in Path(f"assets/augments/{color}").glob("*.{png,jpg,jpeg,webp}")]
    
    # Load augments for different colors
    colors = ["silver", "gold", "prismatic"]
    for color in colors:
        augments_dict[color] = load_augments(color)

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
    
    player_predictions = dict()

    # For each player:
    for player in player_frames:
        player_predictions[player] = dict()
        print(player_predictions)

        # Get image files
        image_files = player_frames[player]
        
        if not image_files:
            print(f"No image files found for {player}")
            continue
        else:
            print(f"Analyzing augment image files for {player}")
        
        print(f"Processing {len(image_files)} images")
        
        # Process each image
        for img_path in image_files:
            base_filename = Path(img_path).stem
            
            # Extract ROI
            full_img, roi = extract_roi(img_path)
            
            if roi is None:
                print(f"  Failed to extract ROI")
                continue
            
            # Run detection on ROI
            detection_results = yolo_model.predict(
                source=roi,
                conf=CONF_THRESHOLD,
                verbose=False
            )
            
            # Check if any boxes detected
            if len(detection_results) > 0 and len(detection_results[0].boxes) > 0:
                boxes = detection_results[0].boxes.data.cpu().numpy()

                # Process up to 3 augments
                augment_limit = min(3, len(boxes))
                boxes = boxes[:augment_limit]
                
                # Split boxes and get sub-images
                all_sub_images = []
                
                for box in boxes:
                    sub_images = split_box(
                        roi, box, base_filename
                    )
                    index = 0
                    for sub_img in sub_images:
                        preds = predict_augment(classifier, sub_img, classes, device, top_k=5)


                        if preds[0]['probability'] < 0.8:
                            print(f"Low confidence for {player}, check: {img_path}")

                        #if preds[0]['class'] not in augments_dict[augments[index]]:
                        #    print(f"Wrong color augment for {player}. {preds[0]['class']} is not in {augments[index]}")
                        
                        if index in player_predictions[player]:
                            player_predictions[player][index].append(preds[0]['class'])
                        else:
                            player_predictions[player][index] = [preds[0]['class']]


                        index += 1

    # accept most frequently tagged augment for each color               
    for player in player_predictions:
        for i in range(0, 3):
            if i in player_predictions[player]:
                player_predictions[player][i] = string_match.most_frequent_string(player_predictions[player][i])

    return player_predictions