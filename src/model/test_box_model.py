import os
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration variables
MODEL_PATH = "runs/box_detection/weights/best.pt"
IMAGES_DIR = "temp/frames"  # Directory with real-world images
OUTPUT_DIR = "real_world_results"
ROI_X = 1270
ROI_Y = 220
ROI_WIDTH = 160
ROI_HEIGHT = 160
CONF_THRESHOLD = 0.25
SHOW_FULL_IMAGE = True  # Set to True to show detection on full image with ROI highlighted

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

def draw_on_full_image(full_img, detection=None):
    """Draw ROI and detection boxes on full image"""
    # Create a copy of the image
    result_img = full_img.copy()
    
    # Draw ROI rectangle
    cv2.rectangle(result_img, (ROI_X, ROI_Y), (ROI_X+ROI_WIDTH, ROI_Y+ROI_HEIGHT), (0, 255, 0), 2)
    
    # Draw detection if available
    if detection is not None and len(detection) > 0:
        for det in detection:
            # Extract bounding box
            x1, y1, x2, y2, conf, cls = det
            
            # Convert coordinates to full image scale
            x1 = int(x1 + ROI_X)
            y1 = int(y1 + ROI_Y)
            x2 = int(x2 + ROI_X)
            y2 = int(y2 + ROI_Y)
            
            # Draw box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label with confidence
            label = f"Box: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_img, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), (0, 0, 255), -1)
            cv2.putText(result_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_img

def test_on_images():
    """Test model on real-world images"""
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
    
    print(f"Testing on {len(image_files)} images")
    
    # Results tracking
    results = {
        "total_images": len(image_files),
        "detections": 0,
        "confidences": []
    }
    
    # Process each image
    for img_path in tqdm(image_files):
        # Extract ROI
        full_img, roi = extract_roi(img_path)
        
        if roi is None:
            continue
        
        # Save ROI for reference
        roi_filename = f"roi_{os.path.basename(img_path)}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, roi_filename), roi)
        
        # Run detection
        detection_results = model.predict(
            source=roi,
            conf=CONF_THRESHOLD,
            verbose=False
        )
        
        # Get boxes
        if len(detection_results) > 0 and len(detection_results[0].boxes) > 0:
            boxes = detection_results[0].boxes
            results["detections"] += 1
            
            # Store confidences
            for conf in boxes.conf.cpu().numpy():
                results["confidences"].append(float(conf))
            
            # Create visualization
            if SHOW_FULL_IMAGE:
                # Draw on full image
                result_img = draw_on_full_image(
                    full_img,
                    detection=boxes.data.cpu().numpy()
                )
            else:
                # Use YOLO's built-in visualization on ROI only
                result_img = detection_results[0].plot()
        else:
            # No detection
            if SHOW_FULL_IMAGE:
                result_img = draw_on_full_image(full_img)
            else:
                result_img = roi
        
        # Save result
        result_filename = f"result_{os.path.basename(img_path)}"
        if SHOW_FULL_IMAGE:
            # Convert from BGR to RGB for saving
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Save as RGB
        Image.fromarray(result_img).save(os.path.join(OUTPUT_DIR, result_filename))
    
    # Output statistics
    print(f"\nResults Summary:")
    print(f"Total images processed: {results['total_images']}")
    print(f"Images with detections: {results['detections']} ({results['detections']/results['total_images']*100:.1f}%)")
    
    if results['confidences']:
        print(f"Average confidence: {np.mean(results['confidences']):.3f}")
        print(f"Min confidence: {np.min(results['confidences']):.3f}")
        print(f"Max confidence: {np.max(results['confidences']):.3f}")
    
    # Plot confidence distribution if detections exist
    if results['confidences']:
        plt.figure(figsize=(10, 6))
        plt.hist(results['confidences'], bins=20, alpha=0.7)
        plt.title('Detection Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_distribution.png'))
        print(f"Saved confidence distribution to {OUTPUT_DIR}/confidence_distribution.png")

if __name__ == "__main__":
    test_on_images()