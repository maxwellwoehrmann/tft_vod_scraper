from ultralytics import YOLO
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import shutil
from typing import Tuple

class TFTAugmentDetector:
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        metadata_path: str,
        classes_path: str,
        board_size: Tuple[int, int] = (130, 100)  # Add board_size parameter with default
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.metadata_path = Path(metadata_path)
        self.classes_path = Path(classes_path)
        self.board_size = board_size  # Store board_size
        
        self.dataset_dir = self.output_dir / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        with open(classes_path) as f:
            self.classes = [line.strip() for line in f]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _create_yolo_label(self, image_name: str, label_dir: Path):
        """Create YOLO format label file using individual augment positions."""
        metadata = self.metadata[image_name]
        augments = metadata['augments']
        
        image_width, image_height = self.board_size
        
        with open(label_dir / f"{Path(image_name).stem}.txt", 'w') as f:
            for augment in augments:
                # Calculate normalized coordinates for each augment
                x_center = (augment['x'] + augment['width'] / 2) / image_width
                y_center = (augment['y'] + augment['height'] / 2) / image_height
                width = augment['width'] / image_width
                height = augment['height'] / image_height
                
                class_idx = self.class_to_idx[augment['name']]
                f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")

    def prepare_yolo_dataset(self, train_size=0.8):
        """Prepare dataset in YOLO format."""
        train_dir = self.dataset_dir / "train"
        val_dir = self.dataset_dir / "val"
        
        for d in [train_dir, val_dir]:
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)
        
        images = list(self.metadata.keys())
        train_images, val_images = train_test_split(
            images, train_size=train_size, random_state=42
        )
        
        for img_list, target_dir in [(train_images, train_dir), (val_images, val_dir)]:
            for img in img_list:
                shutil.copy(self.data_dir / img, target_dir / "images" / img)
                self._create_yolo_label(img, target_dir / "labels")
        
        self._create_yaml_config()

    def _create_yaml_config(self):
        """Create YAML configuration file."""
        config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {idx: name for idx, name in enumerate(self.classes)},
            'nc': len(self.classes)
        }
        
        with open(self.dataset_dir / 'dataset.yaml', 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)

    def train_model(self, epochs=50, batch_size=16):
        model = YOLO('yolov8n.pt')
        model.train(
            data=str(self.dataset_dir / 'dataset.yaml'),
            epochs=epochs,
            batch=32,
            imgsz=160,
            project=str(self.output_dir),
            name='tft_augment_detector',
            device='mps',
            verbose=True,
            conf=0.25,
            iou=0.45,
            lr0=0.01,  # Initial learning rate
            lrf=0.001,  # Final learning rate
            cos_lr=True,  # Enable cosine LR scheduler
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            val=True,
            max_det=10
        )

def main():
    detector = TFTAugmentDetector(
        data_dir="assets/generated_training",
        output_dir="model_output",
        metadata_path="assets/generated_training/metadata.json",
        classes_path="assets/augment_classes.txt"
    )
    
    detector.prepare_yolo_dataset(train_size=0.8)
    detector.train_model(epochs=100, batch_size=32)

if __name__ == "__main__":
    main()