from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import glob
import os

class TFTAugmentPredictor:
    def __init__(self, model_path: str, classes_path: str):
        """
        Initialize predictor with trained model and class names.
        
        Args:
            model_path: Path to the trained YOLO model weights
            classes_path: Path to text file containing class names
        """
        self.model = YOLO(model_path)
        
        # Load class names
        with open(classes_path) as f:
            self.classes = [line.strip() for line in f]
    
    def predict_image(self, image_path: str, conf_threshold: float = 0.25):
        """
        Run prediction on a single image.
        
        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold for detections
        """
        # Run prediction
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            device='mps'
        )
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            detections = []
            for box in boxes:
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = self.classes[class_id]
                confidence = float(box.conf[0])
                
                # Get coordinates (xmin, ymin, xmax, ymax)
                coords = box.xyxy[0].tolist()
                
                detections.append({
                    'class': class_name,
                    'confidence': round(confidence, 3),
                    'box': coords
                })
            
            # Sort by x-coordinate since augments read left to right
            detections.sort(key=lambda x: x['box'][0])
            
            return detections

def save_subsection(image_path, x, y, width, height, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Extract subsection
    subsection = image[y:y+height, x:x+width]
    
    # Resize the subsection
    resized_img = cv2.resize(subsection, (130, 100), interpolation=cv2.INTER_LANCZOS4)
    
    # Save the resized subsection
    cv2.imwrite(output_path, resized_img)


def main():
    os.makedirs('model_output/test_images', exist_ok=True)
    
    # Extract subsections from source images
    for jpg_file in glob.glob('temp/frames/*.jpg'):
        save_subsection(jpg_file, 1300, 280, 130, 100, 
                        f'model_output/test_images/{Path(jpg_file).stem}.png')

    # Configuration
    MODEL_PATH = "model_output/tft_augment_detector/weights/best.pt"
    CLASSES_PATH = "assets/augment_classes.txt"
    TEST_IMAGES_DIR = "model_output/test_images"  
    # TEST_IMAGES_DIR = "model_output/dataset/val/images"
    
    # Initialize predictor
    predictor = TFTAugmentPredictor(MODEL_PATH, CLASSES_PATH)
    
    # Process all images in test directory
    test_dir = Path(TEST_IMAGES_DIR)
    for image_path in test_dir.glob('*.png'):
        print(f"\nProcessing {image_path.name}:")
        
        detections = predictor.predict_image(str(image_path))
        
        # Print detections in order
        for i, det in enumerate(detections, 1):
            print(f"{i}. {det['class']} (confidence: {det['confidence']})")


if __name__ == "__main__":
    main()