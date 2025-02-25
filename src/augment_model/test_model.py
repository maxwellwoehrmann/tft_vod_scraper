from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import glob
import os
from PIL import Image

class ImageNormalizer:
    @staticmethod
    def normalize_image(image_path: str) -> np.ndarray:
        """Normalize image brightness and contrast."""
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to float32 for processing
        img_float = img.astype(np.float32)
        
        # Normalize to 0-1 range
        img_norm = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX)
        
        return img_norm.astype(np.uint8)

class TFTAugmentPredictor:
    def __init__(self, model_path: str, classes_path: str):
        """Initialize predictor with trained model and class names."""
        self.model = YOLO(model_path)
        self.normalizer = ImageNormalizer()
        
        # Load class names
        with open(classes_path) as f:
            self.classes = [line.strip() for line in f]
    
    def predict_image(self, image_path: str, conf_threshold: float = 0.25):
        """Run prediction on a single image with normalization."""
        # Normalize the image first
        normalized_image = self.normalizer.normalize_image(image_path)
        
        # Save normalized image to temporary file
        temp_path = 'temp_normalized.png'
        cv2.imwrite(temp_path, normalized_image)
        
        try:
            # Run prediction on normalized image
            results = self.model.predict(
                source=temp_path,
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
                    
                    # Get coordinates
                    coords = box.xyxy[0].tolist()
                    
                    detections.append({
                        'class': class_name,
                        'confidence': round(confidence, 3),
                        'box': coords
                    })
                
                # Sort by x-coordinate
                detections.sort(key=lambda x: x['box'][0])
                
                return detections
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

def save_subsection(image_path, x, y, width, height, output_path):
    """Save normalized subsection of the image."""
    # Read the image
    image = cv2.imread(image_path)
    
    # Extract subsection
    subsection = image[y:y+height, x:x+width]
    
    # Resize the subsection
    resized_img = cv2.resize(subsection, (130, 100), interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize the image
    normalized = cv2.normalize(resized_img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    
    # Save the normalized and resized subsection
    cv2.imwrite(output_path, normalized.astype(np.uint8))

def main():
    os.makedirs('test_images', exist_ok=True)
    
    # Extract and normalize subsections from source images
    for jpg_file in glob.glob('temp/frames/*.jpg'):
        save_subsection(jpg_file, 1300, 280, 130, 100, 
                       f'test_images/{Path(jpg_file).stem}.png')

    # Configuration
    MODEL_PATH = "model_output/tft_augment_detector/weights/best.pt"
    CLASSES_PATH = "assets/augment_classes.txt"
    TEST_IMAGES_DIR = "test_images"
    
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