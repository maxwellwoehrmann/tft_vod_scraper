import os
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

# Configuration variables
MODEL_PATH = "runs/box_detection/weights/best.pt"
IMAGES_DIR = "temp/frames"  # Directory with images
OUTPUT_DIR = "split_boxes"
ROI_X = 1270
ROI_Y = 220
ROI_WIDTH = 160
ROI_HEIGHT = 160
CONF_THRESHOLD = 0.25
SUB_IMAGE_WIDTH = 30
SUB_IMAGE_HEIGHT = 30

def extract_roi(image_path):
    """Extract region of interest from a full-sized image"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None, None
    
    # Extract ROI
    roi = img[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
    
    # Check if ROI is valid
    if roi.shape[0] != ROI_HEIGHT or roi.shape[1] != ROI_WIDTH:
        print(f"Warning: Extracted ROI dimensions {roi.shape} don't match expected {ROI_WIDTH}x{ROI_HEIGHT}")
    
    return img, roi

def split_and_save_box(roi, box, base_filename, output_dir):
    """
    Split detected box into 30x30 sub-images and save them
    
    Args:
        roi: The region of interest image
        box: Detection box coordinates [x1, y1, x2, y2, conf, cls]
        base_filename: Base filename for saving
        output_dir: Directory to save split images
    
    Returns:
        number of sub-images created
    """
    # Extract box coordinates
    x1, y1, x2, y2, conf, cls = box
    
    # Convert to integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Extract the box image
    box_img = roi[y1:y2, x1:x2]
    
    # Check dimensions
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Debug info
    # print(f"Box dimensions: {box_width}x{box_height}, Confidence: {conf:.2f}")
    
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
    
    # Save all sub-images
    for i, sub_img in enumerate(sub_images):
        # Create filename: original_0001_box1_part1.jpg
        sub_filename = f"{base_filename}_box{int(conf*100):03d}_part{i+1}.jpg"
        output_path = os.path.join(output_dir, sub_filename)
        
        # Convert from BGR to RGB for saving with PIL
        sub_img_rgb = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
        Image.fromarray(sub_img_rgb).save(output_path)
    
    return len(sub_images)

def process_images():
    """Process images, detect boxes, split and save sub-images"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print(f"Loading model from {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set up device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get image files
    image_files = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in {IMAGES_DIR}")
        return
    
    print(f"Processing {len(image_files)} images")
    
    # Results tracking
    results = {
        "total_images": len(image_files),
        "images_with_detections": 0,
        "total_detections": 0,
        "total_sub_images": 0,
        "sub_images_by_split": {1: 0, 2: 0, 3: 0, 4: 0}  # Count by number of splits
    }
    
    # Process each image
    for img_path in tqdm(image_files):
        # Get base filename without extension
        base_filename = Path(img_path).stem
        
        # Extract ROI
        _, roi = extract_roi(img_path)
        
        if roi is None:
            continue
        
        # Run detection
        detection_results = model.predict(
            source=roi,
            conf=CONF_THRESHOLD,
            verbose=False
        )
        
        # Process detections
        if len(detection_results) > 0 and len(detection_results[0].boxes) > 0:
            boxes = detection_results[0].boxes.data.cpu().numpy()
            
            results["images_with_detections"] += 1
            results["total_detections"] += len(boxes)
            
            # Process each detection
            for i, box in enumerate(boxes):
                # Split and save
                num_sub_images = split_and_save_box(roi, box, base_filename, OUTPUT_DIR)
                
                # Update statistics
                results["total_sub_images"] += num_sub_images
                
                # Track split counts (1, 2, 3 splits)
                if num_sub_images in results["sub_images_by_split"]:
                    results["sub_images_by_split"][num_sub_images] += 1
                else:
                    results["sub_images_by_split"][num_sub_images] = 1
    
    # Output statistics
    print("\nResults Summary:")
    print(f"Total images processed: {results['total_images']}")
    print(f"Images with detections: {results['images_with_detections']} ({results['images_with_detections']/results['total_images']*100:.1f}%)")
    print(f"Total boxes detected: {results['total_detections']}")
    print(f"Total sub-images created: {results['total_sub_images']}")
    print("\nSub-images by split count:")
    for split_count, count in sorted(results["sub_images_by_split"].items()):
        print(f"  {split_count} part splits: {count} boxes")
    
    print(f"\nAll sub-images have been saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    process_images()