from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import os
import torch

class AugmentRegionDetector:
    """Detects the black region containing TFT augments within a fixed ROI."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        device: str = 'mps'
    ):
        """
        Initialize the region detector.
        
        Args:
            model_path: Path to pretrained model, or None to use a new model
            conf_threshold: Confidence threshold for detections
            device: Device to run the model on ('cuda', 'cpu', or 'mps')
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded region detector from {model_path}")
        else:
            self.model = YOLO('yolov8n.pt')
            print("Created new region detector from YOLOv8n base model")
            
        self.conf_threshold = conf_threshold
        self.device = device
        
    def train(
        self,
        data_yaml_path: str,
        epochs: int = 50,
        batch_size: int = 32,
        image_size: int = 160,  # Slightly larger than 130x100 for context
        output_dir: str = "model_output/region_detector"
    ) -> str:
        """
        Train the region detector model.
        
        Args:
            data_yaml_path: Path to YAML file with dataset configuration
            epochs: Number of training epochs
            batch_size: Batch size for training
            image_size: Input image size
            output_dir: Directory to save model outputs
            
        Returns:
            Path to the trained model weights
        """
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            project=os.path.dirname(output_dir),
            name=os.path.basename(output_dir),
            device=self.device,
            verbose=True,
            # Learning rate parameters optimized for fast convergence
            lr0=0.01,
            lrf=0.001,
            cos_lr=True,
            # Warmup parameters
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
        )
        
        # Return path to best weights
        weights_path = os.path.join(output_dir, "weights/best.pt")
        return weights_path if os.path.exists(weights_path) else None
    
    def detect_regions(
        self, 
        image: Union[str, np.ndarray],
        roi: Tuple[int, int, int, int] = (1300, 280, 130, 100)  # Default ROI coordinates
    ) -> List[Dict[str, Union[np.ndarray, float]]]:
        """
        Detect augment regions in an image within the specified ROI.
        
        Args:
            image: Path to image or numpy array
            roi: Region of interest as (x, y, width, height)
            
        Returns:
            List of dictionaries, each containing:
                - 'crop': Cropped image containing the augment region
                - 'box': [x1, y1, x2, y2] coordinates of the region relative to ROI
                - 'confidence': Detection confidence
        """
        # Load image
        if isinstance(image, str):
            orig_img = cv2.imread(image)
        else:
            orig_img = image.copy()
        
        # Extract ROI
        x, y, w, h = roi
        roi_img = orig_img[y:y+h, x:x+w]
        
        # Ensure ROI is valid
        if roi_img.size == 0:
            print(f"Warning: Invalid ROI {roi} for image shape {orig_img.shape}")
            return []
        
        # Run inference on the ROI
        results = self.model.predict(
            source=roi_img,
            conf=self.conf_threshold,
            device=self.device
        )
        
        regions = []
        
        # Process results
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get coordinates (xmin, ymin, xmax, ymax) within the ROI
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Crop region from ROI
                    region_crop = roi_img[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Add to regions list
                    regions.append({
                        'crop': region_crop,
                        'box': [int(x1), int(y1), int(x2), int(y2)],  # Coordinates relative to ROI
                        'box_absolute': [int(x+x1), int(y+y1), int(x+x2), int(y+y2)],  # Absolute coordinates
                        'confidence': float(box.conf[0])
                    })
        
        return regions
    
    def save_regions(
        self,
        image_path: str,
        output_dir: str = "detected_regions",
        roi: Tuple[int, int, int, int] = (1300, 280, 130, 100)
    ) -> List[str]:
        """
        Detect and save cropped regions from an image.
        
        Args:
            image_path: Path to image
            output_dir: Directory to save cropped regions
            roi: Region of interest as (x, y, width, height)
            
        Returns:
            List of paths to saved crops
        """
        os.makedirs(output_dir, exist_ok=True)
        
        regions = self.detect_regions(image_path, roi)
        saved_paths = []
        
        for i, region in enumerate(regions):
            output_path = os.path.join(
                output_dir, 
                f"{Path(image_path).stem}_region_{i}.png"
            )
            cv2.imwrite(output_path, region['crop'])
            saved_paths.append(output_path)
            
        return saved_paths