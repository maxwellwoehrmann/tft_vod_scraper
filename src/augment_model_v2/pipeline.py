from pathlib import Path
import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import shutil
from .region_detector import AugmentRegionDetector
from .data_generator import RegionDataGenerator

class AugmentDetectionPipeline:
    """Pipeline orchestrating the two-stage augment detection process."""
    
    def __init__(
        self,
        region_model_path: Optional[str] = None,
        augment_model_path: Optional[str] = None,
        output_dir: str = "augment_detection_output",
        device: str = 'mps',
        roi: Tuple[int, int, int, int] = (1300, 280, 130, 100)  # Default ROI (x, y, w, h)
    ):
        """
        Initialize the pipeline.
        
        Args:
            region_model_path: Path to trained region detector model, or None
            augment_model_path: Path to trained augment classifier model, or None
            output_dir: Directory for output data
            device: Device to run models on
            roi: Region of interest as (x, y, width, height)
        """
        # Initialize the region detector
        self.region_detector = AugmentRegionDetector(
            model_path=region_model_path,
            device=device
        )
        
        # For now, just set augment_model_path as an attribute
        # We'll implement the augment classifier in the next phase
        self.augment_model_path = augment_model_path
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.roi = roi
        
    def train_region_detector(
        self,
        synthetic_config: Dict,
        epochs: int = 50,
        force_regenerate: bool = False
    ) -> str:
        """
        Train the region detector model.
        
        Args:
            synthetic_config: Configuration for synthetic data generation
            epochs: Number of training epochs
            force_regenerate: Whether to regenerate data even if it exists
            
        Returns:
            Path to trained model weights
        """
        # Create synthetic data generator
        data_dir = Path(synthetic_config.get('output_dir', 'region_detector_data'))
        
        # Check if data already exists
        if force_regenerate or not (data_dir / 'dataset.yaml').exists():
            print("Generating synthetic data for region detector...")
            generator = RegionDataGenerator(
                augments_dir=synthetic_config['augments_dir'],
                boards_dir=synthetic_config['boards_dir'],
                output_dir=str(data_dir),
                num_samples=synthetic_config.get('num_samples', 10000),
                augment_size=synthetic_config.get('augment_size', (30, 30)),
                strip_spacing=synthetic_config.get('strip_spacing', 1),
                roi_size=synthetic_config.get('roi_size', (130, 100))
            )
            generator.generate_dataset()
        else:
            print(f"Using existing data at {data_dir}")
        
        # Train the region detector
        print("Training region detector model...")
        model_path = self.region_detector.train(
            data_yaml_path=str(data_dir / 'dataset.yaml'),
            epochs=epochs,
            output_dir=str(self.output_dir / "region_detector"),
            image_size=160  # Slightly larger than 130x100 for context
        )
        
        return model_path
    
    def process_image(
        self, 
        image_path: str,
        save_crops: bool = True
    ) -> List[Dict]:
        """
        Process an image to detect augment regions.
        
        Args:
            image_path: Path to image
            save_crops: Whether to save cropped regions
            
        Returns:
            List of detection results
        """
        # Detect regions
        regions = self.region_detector.detect_regions(image_path, self.roi)
        
        if save_crops and regions:
            crops_dir = self.output_dir / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each crop
            for i, region in enumerate(regions):
                crop_path = crops_dir / f"{Path(image_path).stem}_region_{i}.png"
                cv2.imwrite(str(crop_path), region['crop'])
                
                # Add path to region info
                region['crop_path'] = str(crop_path)
        
        return regions
    
    def process_directory(
        self, 
        input_dir: str,
        output_subdir: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_subdir: Subdirectory name for outputs (default: timestamp)
            
        Returns:
            Dictionary mapping image paths to detection results
        """
        input_dir = Path(input_dir)
        
        # Create output subdirectory
        if output_subdir:
            output_dir = self.output_dir / output_subdir
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_dir / f"batch_{timestamp}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_paths = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
        
        results = {}
        for img_path in image_paths:
            print(f"Processing {img_path.name}...")
            
            # Process image
            detections = self.process_image(
                str(img_path),
                save_crops=True
            )
            
            # Store results
            results[str(img_path)] = detections
            
            # Visualize and save results
            self._visualize_detections(
                str(img_path),
                detections,
                str(output_dir / f"{img_path.stem}_detected.jpg")
            )
        
        return results
    
    def _visualize_detections(
        self,
        image_path: str,
        detections: List[Dict],
        output_path: str
    ) -> None:
        """
        Visualize detections on an image.
        
        Args:
            image_path: Path to original image
            detections: List of detection results
            output_path: Path to save visualization
        """
        # Read image
        image = cv2.imread(image_path)
        
        # Draw ROI
        x, y, w, h = self.roi
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (255, 200, 0),  # Yellow for ROI
            1
        )
        
        # Draw detected regions
        for i, detection in enumerate(detections):
            # Use absolute coordinates
            box = detection.get('box_absolute', detection['box'])
            confidence = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(
                image,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 255, 0),  # Green for detections
                2
            )
            
            # Draw label
            cv2.putText(
                image,
                f"Region {i+1}: {confidence:.2f}",
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Save image
        cv2.imwrite(output_path, image)