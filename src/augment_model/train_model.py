from ultralytics import YOLO
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import shutil

class TFTAugmentDetector:
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        metadata_path: str,
        classes_path: str
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.metadata_path = Path(metadata_path)
        self.classes_path = Path(classes_path)
        
        # Create necessary directories
        self.dataset_dir = self.output_dir / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata and classes
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        with open(classes_path) as f:
            self.classes = [line.strip() for line in f]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def prepare_yolo_dataset(self, train_size=0.8):
        """Prepare dataset in YOLO format."""
        # Create directory structure
        train_dir = self.dataset_dir / "train"
        val_dir = self.dataset_dir / "val"
        
        for d in [train_dir, val_dir]:
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)
        
        # Split data
        images = list(self.metadata.keys())
        train_images, val_images = train_test_split(
            images, train_size=train_size, random_state=42
        )
        
        # Process images
        for img_list, target_dir in [
            (train_images, train_dir),
            (val_images, val_dir)
        ]:
            for img in img_list:
                # Copy image
                src_img = self.data_dir / img
                dst_img = target_dir / "images" / img
                shutil.copy(src_img, dst_img)
                
                # Create label
                self._create_yolo_label(img, target_dir / "labels")
        
        self._create_yaml_config()

    def _create_yolo_label(self, image_name: str, label_dir: Path):
        """Create YOLO format label file."""
        metadata = self.metadata[image_name]
        augments = metadata['augments']
        panel_origin = metadata['panel_origin']
        
        # Image dimensions from generator
        image_width = 130
        image_height = 100
        
        with open(label_dir / f"{Path(image_name).stem}.txt", 'w') as f:
            for idx, augment in enumerate(augments):
                # Calculate box coordinates
                x_center = (panel_origin['x'] + (idx * 31) + 15) / image_width
                y_center = (panel_origin['y'] + 15) / image_height
                width = 30 / image_width
                height = 30 / image_height
                
                # Write in YOLO format: class x_center y_center width height
                class_idx = self.class_to_idx[augment]
                f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")

    def _create_yaml_config(self):
        """Create YAML configuration file."""
        config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {idx: name for idx, name in enumerate(self.classes)},
            'nc': len(self.classes)
        }
        
        yaml_path = self.dataset_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)

    def train_model(self, epochs=50, batch_size=16):
        """Train YOLOv8 model with improved parameters."""
        # Initialize model - using small instead of nano
        model = YOLO('yolov8s.pt')
        
        # Train
        model.train(
            data=str(self.dataset_dir / 'dataset.yaml'),
            epochs=epochs,
            batch=batch_size,
            imgsz=224,
            project=str(self.output_dir),
            name='tft_augment_detector',
            device='mps',
            plots=True,
            verbose=True,
            lr0=0.01,           # Higher initial learning rate
            max_det=10,         # Max detections per image (never more than 3 augments)
            patience=10         # Early stopping if no improvement
        )

def main():
    # Configuration
    DATA_DIR = "assets/generated_training"
    OUTPUT_DIR = "model_output"
    METADATA_PATH = "assets/generated_training/metadata.json"
    CLASSES_PATH = "assets/augment_classes.txt"
    
    # Create and train detector
    detector = TFTAugmentDetector(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        metadata_path=METADATA_PATH,
        classes_path=CLASSES_PATH
    )
    
    # Prepare dataset
    detector.prepare_yolo_dataset(train_size=0.8)
    
    # Train model
    detector.train_model(epochs=50, batch_size=16)

if __name__ == "__main__":
    main()