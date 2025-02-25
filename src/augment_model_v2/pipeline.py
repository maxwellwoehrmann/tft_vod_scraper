from pathlib import Path
import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import shutil
from .region_detector import AugmentRegionDetector
from .augment_classifier import AugmentClassifier
from .augment_data_generator import AugmentDataGenerator

class AugmentDetectionPipeline:
    """Pipeline orchestrating the two-stage augment detection process."""
    
    def __init__(
        self,
        region_model_path: Optional[str] = None,
        augment_model_path: Optional[str] = None,
        output_dir: str = "augment_detection_output",
        device: str = 'mps',
        roi: Tuple[int, int, int, int] = (1300, 280, 130, 100),
        augment_templates_dir: Optional[str] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            region_model_path: Path to trained region detector model, or None
            augment_model_path: Path to trained augment classifier model, or None
            output_dir: Directory for output data
            device: Device to run models on
            roi: Region of interest as (x, y, width, height)
            augment_templates_dir: Directory containing high-quality augment templates
        """
        # Initialize the region detector
        self.region_detector = AugmentRegionDetector(
            model_path=region_model_path,
            device=device
        )
        
        # Initialize the augment classifier
        self.augment_classifier = AugmentClassifier(
            model_path=augment_model_path,
            device=device,
            augment_templates_dir=augment_templates_dir
        )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.roi = roi
        self.augment_templates_dir = augment_templates_dir
        
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
                roi_size=synthetic_config.get('roi_size', (130, 100)),
                board_crop_coords=synthetic_config.get('board_crop_coords', (380, 130, 440, 170))
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
    
    def train_augment_classifier(
        self,
        augments_dir: str,
        classes_path: str,
        num_samples: int = 20000,
        epochs: int = 50,
        region_crops_dir: Optional[str] = None,
        force_regenerate: bool = False
    ) -> str:
        """
        Train the augment classifier model.
        
        Args:
            augments_dir: Directory containing augment images
            classes_path: Path to file containing class names
            num_samples: Number of synthetic samples to generate
            epochs: Number of training epochs
            region_crops_dir: Directory containing real region crops (optional)
            force_regenerate: Whether to regenerate data even if it exists
            
        Returns:
            Path to trained model weights
        """
        data_dir = self.output_dir / "augment_classifier_data"
        
        # Check if data already exists
        if force_regenerate or not (data_dir / 'dataset.yaml').exists():
            print("Generating synthetic data for augment classifier...")
            generator = AugmentDataGenerator(
                augments_dir=augments_dir,
                output_dir=str(data_dir),
                num_samples=num_samples,
                augment_size=(30, 30),
                crop_size=(130, 100),
                classes_path=classes_path
            )
            generator.generate_dataset()
        else:
            print(f"Using existing data at {data_dir}")
        
        # If we have region crops, prepare a dataset combining synthetic and real data
        if region_crops_dir and Path(region_crops_dir).exists():
            print("Preparing training data from real region crops...")
            real_data_dir = self.output_dir / "augment_real_data"
            yaml_path = self.augment_classifier.prepare_augment_training_data(
                region_crops_dir=region_crops_dir,
                output_dir=str(real_data_dir),
                classes_path=classes_path
            )
            
            # Train on real data first
            print("Training augment classifier on real data...")
            self.augment_classifier.train(
                data_yaml_path=yaml_path,
                epochs=max(10, epochs // 5),  # Shorter training on real data
                output_dir=str(self.output_dir / "augment_classifier_real")
            )
            
            # Then continue with synthetic data
            print("Fine-tuning augment classifier on synthetic data...")
            model_path = self.augment_classifier.train(
                data_yaml_path=str(data_dir / 'dataset.yaml'),
                epochs=epochs,
                output_dir=str(self.output_dir / "augment_classifier")
            )
        else:
            # Train directly on synthetic data
            print("Training augment classifier on synthetic data...")
            model_path = self.augment_classifier.train(
                data_yaml_path=str(data_dir / 'dataset.yaml'),
                epochs=epochs,
                output_dir=str(self.output_dir / "augment_classifier")
            )
        
        return model_path
    
    def process_image(
        self, 
        image_path: str,
        save_crops: bool = True,
        save_results: bool = True
    ) -> List[Dict]:
        """
        Process an image using the two-stage detection pipeline.
        
        Args:
            image_path: Path to image
            save_crops: Whether to save cropped regions
            save_results: Whether to save visualization of results
            
        Returns:
            List of detection results with augment information
        """
        # Stage 1: Detect regions
        regions = self.region_detector.detect_regions(image_path, self.roi)
        
        results = []
        crops_dir = self.output_dir / "crops" if save_crops else None
        
        if save_crops and regions:
            crops_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage 2: Classify augments in each region
        for i, region in enumerate(regions):
            crop = region['crop']
            
            # Save crop if requested
            if save_crops:
                crop_path = crops_dir / f"{Path(image_path).stem}_region_{i}.png"
                cv2.imwrite(str(crop_path), crop)
                region['crop_path'] = str(crop_path)
            
            # Detect augments in the region
            augments = self.augment_classifier.detect_augments(
                crop, 
                verify_with_templates=self.augment_templates_dir is not None
            )
            
            # Add augments to region info
            region['augments'] = augments
            results.append(region)
        
        # Visualize results if requested
        if save_results:
            self._visualize_full_results(
                image_path,
                results,
                str(self.output_dir / f"{Path(image_path).stem}_detected.jpg")
            )
        
        return results
    
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
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        
        # Find all images
        image_paths = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
        
        results = {}
        for img_path in image_paths:
            print(f"Processing {img_path.name}...")
            
            # Process image with both stages
            detections = self.process_image(
                str(img_path),
                save_crops=True,
                save_results=True
            )
            
            # Store results
            results[str(img_path)] = detections
        
        # Save results as JSON
        import json
        
        # Convert results to serializable format
        serializable_results = {}
        for img_path, detections in results.items():
            serializable_detections = []
            for det in detections:
                # Remove numpy arrays and cv2 images
                det_copy = det.copy()
                if 'crop' in det_copy:
                    del det_copy['crop']
                serializable_detections.append(det_copy)
            serializable_results[img_path] = serializable_detections
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return results
    
    def _visualize_full_results(
        self,
        image_path: str,
        results: List[Dict],
        output_path: str
    ) -> None:
        """
        Visualize both region and augment detections on an image.
        
        Args:
            image_path: Path to original image
            results: List of detection results including augments
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
        
        # Draw regions and augments
        for i, detection in enumerate(results):
            # Use absolute coordinates for region
            region_box = detection.get('box_absolute', detection['box'])
            confidence = detection['confidence']
            
            # Draw region rectangle
            cv2.rectangle(
                image,
                (region_box[0], region_box[1]),
                (region_box[2], region_box[3]),
                (0, 255, 0),  # Green for regions
                2
            )
            
            # Draw region label
            cv2.putText(
                image,
                f"Region {i+1}: {confidence:.2f}",
                (region_box[0], region_box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
            # Draw augments if any
            if 'augments' in detection and detection['augments']:
                for j, augment in enumerate(detection['augments']):
                    # Get augment box relative to full image
                    augment_box = augment['box']
                    augment_class = augment['class']
                    augment_conf = augment['confidence']
                    
                    # Adjust box to image coordinates
                    abs_box = [
                        region_box[0] + augment_box[0],
                        region_box[1] + augment_box[1],
                        region_box[0] + augment_box[2],
                        region_box[1] + augment_box[3]
                    ]
                    
                    # Draw augment rectangle
                    cv2.rectangle(
                        image,
                        (int(abs_box[0]), int(abs_box[1])),
                        (int(abs_box[2]), int(abs_box[3])),
                        (0, 0, 255),  # Red for augments
                        2
                    )
                    
                    # Draw augment label
                    cv2.putText(
                        image,
                        f"{augment_class}: {augment_conf:.2f}",
                        (int(abs_box[0]), int(abs_box[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1
                    )
        
        # Save image
        cv2.imwrite(output_path, image)