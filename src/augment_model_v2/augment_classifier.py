from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import os
import torch
import shutil

class AugmentClassifier:
    """Identifies specific augments within detected regions."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        device: str = 'mps',
        augment_templates_dir: Optional[str] = None
    ):
        """
        Initialize the augment classifier.
        
        Args:
            model_path: Path to pretrained model, or None to use a new model
            conf_threshold: Confidence threshold for detections
            device: Device to run the model on ('cuda', 'cpu', or 'mps')
            augment_templates_dir: Directory containing high-quality augment templates for verification
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded augment classifier from {model_path}")
        else:
            self.model = YOLO('yolov8n.pt')
            print("Created new augment classifier from YOLOv8n base model")
            
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Load augment templates for verification if provided
        self.template_features = {}
        if augment_templates_dir:
            self._load_templates(augment_templates_dir)
    
    def _load_templates(self, templates_dir: str):
        """
        Load high-quality augment templates and extract features.
        
        Args:
            templates_dir: Directory containing template images
        """
        templates_dir = Path(templates_dir)
        
        # Create SIFT feature extractor
        self.sift = cv2.SIFT_create()
        
        # Process each template image
        for img_path in templates_dir.glob('*.webp'):
            augment_name = img_path.stem
            
            # Read and preprocess template
            template = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if template is None:
                continue
                
            # Convert to grayscale for feature extraction
            gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Extract keypoints and descriptors
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            if descriptors is not None:
                self.template_features[augment_name] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'image': template
                }
        
        print(f"Loaded {len(self.template_features)} augment templates")
    
    def train(
        self,
        data_yaml_path: str,
        epochs: int = 50,
        batch_size: int = 32,
        image_size: int = 160,
        output_dir: str = "model_output/augment_classifier"
    ) -> str:
        """
        Train the augment classifier model.
        
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
    
    def detect_augments(
        self, 
        region: Union[str, np.ndarray],
        verify_with_templates: bool = True
    ) -> List[Dict]:
        """
        Detect and classify augments in a cropped region.
        
        Args:
            region: Path to region image or numpy array
            verify_with_templates: Whether to verify detections using templates
            
        Returns:
            List of dictionaries, each containing:
                - 'class': Class name of the augment
                - 'confidence': Detection confidence
                - 'box': Bounding box coordinates [x1, y1, x2, y2]
        """
        # Run inference
        results = self.model.predict(
            source=region,
            conf=self.conf_threshold,
            device=self.device
        )
        
        detections = []
        
        # Process results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                for box in boxes:
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Get coordinates (xmin, ymin, xmax, ymax)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'box': [float(x1), float(y1), float(x2), float(y2)]
                    }
                    
                    # If region is an image array and template verification is enabled
                    if verify_with_templates and isinstance(region, np.ndarray) and self.template_features:
                        # Extract the detected augment
                        augment_img = region[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Verify with templates
                        verified_class, similarity = self._verify_with_templates(augment_img)
                        
                        # If verification is confident and differs from detection
                        if similarity > 0.6 and verified_class != class_name:
                            detection['original_class'] = class_name
                            detection['class'] = verified_class
                            detection['template_similarity'] = similarity
                    
                    detections.append(detection)
        
        # Sort detections by x-coordinate (left to right)
        if detections:
            detections.sort(key=lambda x: x['box'][0])
        
        return detections
    
    def _verify_with_templates(self, augment_img: np.ndarray) -> Tuple[str, float]:
        """
        Verify a detected augment against high-quality templates.
        
        Args:
            augment_img: Cropped augment image
            
        Returns:
            Tuple of (best_match_class, similarity_score)
        """
        if not self.template_features:
            return None, 0.0
            
        # Convert to grayscale
        gray = cv2.cvtColor(augment_img, cv2.COLOR_BGR2GRAY)
        
        # Extract features
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 2:
            return None, 0.0
        
        # FLANN parameters for matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        best_match = None
        best_score = 0.0
        
        # Compare with each template
        for class_name, template_data in self.template_features.items():
            template_descriptors = template_data['descriptors']
            
            if len(descriptors) < 2 or len(template_descriptors) < 2:
                continue
                
            # Find matches
            matches = flann.knnMatch(descriptors, template_descriptors, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            # Calculate similarity score
            similarity = len(good_matches) / max(len(keypoints), len(template_data['keypoints']))
            
            if similarity > best_score:
                best_score = similarity
                best_match = class_name
        
        return best_match, best_score

    def prepare_augment_training_data(
        self,
        region_crops_dir: str,
        output_dir: str,
        classes_path: str
    ) -> str:
        """
        Prepare training data for the augment classifier from region crops.
        
        Args:
            region_crops_dir: Directory containing cropped regions from region detector
            output_dir: Output directory for training data
            classes_path: Path to text file with augment class names
            
        Returns:
            Path to generated dataset YAML file
        """
        # Create output directories
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        
        # Create train/val split
        train_images_dir = output_dir / "train" / "images"
        train_labels_dir = output_dir / "train" / "labels"
        val_images_dir = output_dir / "val" / "images"
        val_labels_dir = output_dir / "val" / "labels"
        
        for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load class names
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Map class names to indices
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        
        # Process crops
        crops_dir = Path(region_crops_dir)
        crop_paths = list(crops_dir.glob('*.png')) + list(crops_dir.glob('*.jpg'))
        
        print(f"Found {len(crop_paths)} region crops")
        
        # Shuffle and split (80% train, 20% val)
        import random
        random.shuffle(crop_paths)
        split_idx = int(0.8 * len(crop_paths))
        train_crops = crop_paths[:split_idx]
        val_crops = crop_paths[split_idx:]
        
        # Process training crops
        for i, crop_path in enumerate(train_crops):
            # Run initial detection to get augment positions
            detections = self.detect_augments(str(crop_path), verify_with_templates=False)
            
            if not detections:
                continue
                
            # Copy image
            shutil.copy(crop_path, train_images_dir / crop_path.name)
            
            # Create YOLO format label
            with open(train_labels_dir / f"{crop_path.stem}.txt", 'w') as f:
                img = cv2.imread(str(crop_path))
                img_height, img_width = img.shape[:2]
                
                for det in detections:
                    # Get class index (default to 0 if not found)
                    class_idx = class_to_idx.get(det['class'], 0)
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = det['box']
                    
                    # Convert to YOLO format (x_center, y_center, width, height) normalized
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")
        
        # Process validation crops
        for i, crop_path in enumerate(val_crops):
            # Same as training, but for validation set
            detections = self.detect_augments(str(crop_path), verify_with_templates=False)
            
            if not detections:
                continue
                
            shutil.copy(crop_path, val_images_dir / crop_path.name)
            
            with open(val_labels_dir / f"{crop_path.stem}.txt", 'w') as f:
                img = cv2.imread(str(crop_path))
                img_height, img_width = img.shape[:2]
                
                for det in detections:
                    class_idx = class_to_idx.get(det['class'], 0)
                    x1, y1, x2, y2 = det['box']
                    
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")
        
        # Create dataset.yaml for YOLO training
        dataset_config = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(classes),
            'names': {i: name for i, name in enumerate(classes)}
        }
        
        yaml_path = output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Prepared {len(train_crops)} training samples and {len(val_crops)} validation samples")
        return str(yaml_path)