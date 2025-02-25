from ultralytics import YOLO
import numpy as np
from pathlib import Path
import shutil
from typing import List, Dict, Tuple
import json
import cv2
import random

class SelfTrainingManager:
    def __init__(
        self,
        base_model_path: str,
        unlabeled_data_dir: str,
        confidence_threshold: float = 0.85,
        agreement_threshold: float = 0.85
    ):
        self.base_model = YOLO(base_model_path)
        self.unlabeled_dir = Path(unlabeled_data_dir)
        self.conf_threshold = confidence_threshold
        self.agreement_threshold = agreement_threshold
        
        # Create working directories
        self.pseudo_labeled_dir = Path('pseudo_labeled_data')
        self.pseudo_labeled_dir.mkdir(exist_ok=True)
        
    def generate_ensemble_predictions(
        self, 
        image_path: str,
        num_augmentations: int = 5
    ) -> List[Dict]:
        """Generate predictions using test-time augmentation ensemble."""
        predictions = []
        
        # Load image
        image = cv2.imread(image_path)
        
        # Generate predictions with different augmentations
        for _ in range(num_augmentations):
            # Apply random augmentation
            augmented = self._apply_test_augmentation(image)
            
            # Get prediction
            results = self.base_model.predict(
                source=augmented,
                conf=self.conf_threshold,
                device='mps'
            )
            
            # Store detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    predictions.append({
                        'class_id': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'box': box.xyxy[0].tolist()
                    })
        
        return predictions
    
    def _apply_test_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentation for test-time augmentation."""
        aug_image = image.copy()
        
        # Random horizontal flip
        if random.random() < 0.5:
            aug_image = cv2.flip(aug_image, 1)
        
        # Random brightness/contrast
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.uniform(-10, 10)    # Brightness
        aug_image = cv2.convertScaleAbs(aug_image, alpha=alpha, beta=beta)
        
        # Random Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 5, aug_image.shape)
            aug_image = np.clip(aug_image + noise, 0, 255).astype(np.uint8)
        
        return aug_image
    
    def _filter_consistent_predictions(
        self,
        predictions: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """Filter predictions that are consistent across augmentations."""
        filtered = []
        
        # Group predictions by class
        class_predictions = {}
        for pred in predictions:
            class_id = pred['class_id']
            if class_id not in class_predictions:
                class_predictions[class_id] = []
            class_predictions[class_id].append(pred)
        
        # For each class, find consistent predictions
        for class_id, preds in class_predictions.items():
            # Sort by confidence
            preds.sort(key=lambda x: x['confidence'], reverse=True)
            
            while preds:
                base_pred = preds[0]
                matches = 1
                non_matches = 0
                
                # Compare with other predictions
                i = 1
                while i < len(preds):
                    if self._calculate_iou(base_pred['box'], preds[i]['box']) > iou_threshold:
                        matches += 1
                        preds.pop(i)
                    else:
                        non_matches += 1
                        i += 1
                
                # Calculate agreement ratio
                agreement = matches / (matches + non_matches) if matches + non_matches > 0 else 0
                
                # If prediction is consistent enough, keep it
                if agreement > self.agreement_threshold:
                    filtered.append(base_pred)
                
                preds.pop(0)
        
        return filtered
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def generate_pseudo_labels(self):
        """Generate pseudo-labels for unlabeled data."""
        for image_path in self.unlabeled_dir.glob('*.png'):
            # Generate ensemble predictions
            predictions = self.generate_ensemble_predictions(str(image_path))
            
            # Filter consistent predictions
            filtered_predictions = self._filter_consistent_predictions(predictions)
            
            # If we have high-confidence predictions, create label
            if filtered_predictions:
                # Copy image
                shutil.copy(image_path, self.pseudo_labeled_dir / image_path.name)
                
                # Create YOLO format label file
                self._create_yolo_label(
                    image_path.stem,
                    filtered_predictions,
                    self.pseudo_labeled_dir
                )
    
    def _create_yolo_label(
        self,
        image_name: str,
        predictions: List[Dict],
        output_dir: Path
    ):
        """Create YOLO format label file from predictions."""
        # Assuming image dimensions from your training setup
        image_width, image_height = 130, 100
        
        with open(output_dir / f"{image_name}.txt", 'w') as f:
            for pred in predictions:
                # Convert to YOLO format
                box = pred['box']
                x_center = (box[0] + box[2]) / 2 / image_width
                y_center = (box[1] + box[3]) / 2 / image_height
                width = (box[2] - box[0]) / image_width
                height = (box[3] - box[1]) / image_height
                
                f.write(f"{pred['class_id']} {x_center} {y_center} {width} {height}\n")

    def fine_tune_model(self, epochs: int = 20):
        """Fine-tune model on pseudo-labeled data."""
        # Create dataset.yaml for pseudo-labeled data
        dataset_config = {
            'path': str(self.pseudo_labeled_dir.absolute()),
            'train': '.',  # All pseudo-labeled data used for training
            'val': '.',    # Use same data for validation
            'nc': len(self.base_model.names),
            'names': self.base_model.names
        }
        
        with open(self.pseudo_labeled_dir / 'dataset.yaml', 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        # Fine-tune with conservative learning rate
        self.base_model.train(
            data=str(self.pseudo_labeled_dir / 'dataset.yaml'),
            epochs=epochs,
            batch=16,
            imgsz=160,
            device='mps',
            verbose=True,
            conf=self.conf_threshold,
            iou=0.45,
            lr0=0.001,  # Lower learning rate for fine-tuning
            lrf=0.0001,
            cos_lr=True,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=2.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            val=True,
            max_det=10
        )