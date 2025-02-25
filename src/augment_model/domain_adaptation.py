import os
from pathlib import Path
from typing import Optional
import json
import shutil
from ultralytics import YOLO
from train_model import TFTAugmentDetector
from generate_training import TFTTrainingDataGenerator
from self_trainer import SelfTrainingManager

class DomainAdaptationPipeline:
    def __init__(
        self,
        synthetic_data_config: dict,
        real_data_dir: str,
        output_dir: str,
        initial_model_path: Optional[str] = None
    ):
        self.synthetic_config = synthetic_data_config
        self.real_data_dir = Path(real_data_dir)
        self.output_dir = Path(output_dir)
        self.initial_model_path = initial_model_path
        
        # Create working directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_dir = self.output_dir / "synthetic_data"
        self.synthetic_dir.mkdir(exist_ok=True)
        
    def run_pipeline(self):
        """Execute the complete domain adaptation pipeline."""
        print("Starting domain adaptation pipeline...")
        
        # Step 1: Generate enhanced synthetic data
        print("Generating enhanced synthetic data...")
        generator = TFTTrainingDataGenerator(
            augments_dir=self.synthetic_config['augments_dir'],
            boards_dir=self.synthetic_config['boards_dir'],
            output_dir=str(self.synthetic_dir),
            num_samples=self.synthetic_config['num_samples'],
            augment_size=self.synthetic_config['augment_size'],
            strip_spacing=self.synthetic_config['strip_spacing'],
            board_size=self.synthetic_config['board_size'],
            size_variance=self.synthetic_config['size_variance']
        )
        generator.generate_dataset()
        
        # Step 2: Train initial model if not provided
        if not self.initial_model_path:
            print("Training initial model on synthetic data...")
            detector = TFTAugmentDetector(
                data_dir=str(self.synthetic_dir),
                output_dir=str(self.output_dir / "initial_model"),
                metadata_path=str(self.synthetic_dir / "metadata.json"),
                classes_path=self.synthetic_config['classes_path']
            )
            detector.prepare_yolo_dataset(train_size=0.8)
            detector.train_model(epochs=40)  # Reduced epochs for initial training
            self.initial_model_path = str(self.output_dir / "initial_model" / "tft_augment_detector" / "weights" / "best.pt")
        
        # Step 3: Self-training with real data
        print("Starting self-training process...")
        self_trainer = SelfTrainingManager(
            base_model_path=self.initial_model_path,
            unlabeled_data_dir=self.real_data_dir,
            confidence_threshold=0.85,  # High confidence threshold
            agreement_threshold=0.85
        )
        
        # Generate pseudo-labels
        print("Generating pseudo-labels for real data...")
        self_trainer.generate_pseudo_labels()
        
        # Fine-tune on pseudo-labeled data
        print("Fine-tuning model on pseudo-labeled data...")
        self_trainer.fine_tune_model(epochs=10)
        
        # Step 4: Final training combining both datasets
        print("Training final model on combined dataset...")
        self._train_combined_model()
        
        print("Domain adaptation pipeline complete!")
    
    def _train_combined_model(self):
        """Train final model on combined synthetic and pseudo-labeled data."""
        combined_dir = self.output_dir / "combined_data"
        combined_dir.mkdir(exist_ok=True)
        
        # Create directory structure
        train_dir = combined_dir / "train"
        val_dir = combined_dir / "val"
        for d in [train_dir, val_dir]:
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)
        
        # Copy synthetic data
        synthetic_metadata = json.load(open(self.synthetic_dir / "metadata.json"))
        synthetic_files = list(synthetic_metadata.keys())
        
        # Use 80% of synthetic data for training
        train_synthetic = synthetic_files[:int(len(synthetic_files) * 0.8)]
        val_synthetic = synthetic_files[int(len(synthetic_files) * 0.8):]
        
        # Copy synthetic data to appropriate directories
        for img in train_synthetic:
            shutil.copy(self.synthetic_dir / img, train_dir / "images" / img)
            # Changed path to get labels from initial model dataset
            shutil.copy(
                self.output_dir / "initial_model" / "dataset" / "train" / "labels" / f"{Path(img).stem}.txt", 
                train_dir / "labels" / f"{Path(img).stem}.txt"
            )

        for img in val_synthetic:
            shutil.copy(self.synthetic_dir / img, val_dir / "images" / img)
            # Changed path to get labels from initial model dataset
            shutil.copy(
                self.output_dir / "initial_model" / "dataset" / "val" / "labels" / f"{Path(img).stem}.txt",
                val_dir / "labels" / f"{Path(img).stem}.txt"
        )
        # Copy pseudo-labeled data (all to training)
        pseudo_labeled_dir = Path('pseudo_labeled_data')
        for img in pseudo_labeled_dir.glob('*.png'):
            shutil.copy(img, train_dir / "images" / img.name)
            shutil.copy(pseudo_labeled_dir / f"{img.stem}.txt", train_dir / "labels" / f"{img.stem}.txt")
        
        # Create dataset config
        dataset_config = {
            'path': str(combined_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len([line.strip() for line in open(self.synthetic_config['classes_path'])]),
            'names': {i: name.strip() for i, name in enumerate(open(self.synthetic_config['classes_path']))}
        }
        
        with open(combined_dir / 'dataset.yaml', 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        # Train final model
        model = YOLO(self.initial_model_path)
        model.train(
            data=str(combined_dir / 'dataset.yaml'),
            epochs=10,
            batch=32,
            imgsz=160,
            device='mps',
            verbose=True,
            conf=0.25,
            iou=0.45,
            lr0=0.005,
            lrf=0.0005,
            cos_lr=True,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            val=True,
            max_det=10
        )

def main():
    # Configuration
    synthetic_config = {
        'augments_dir': "assets/augments",
        'boards_dir': "assets/boards",
        'num_samples': 20000,
        'augment_size': (30, 30),
        'strip_spacing': 1,
        'board_size': (130, 100),
        'size_variance': 0.15,
        'classes_path': "assets/augment_classes.txt",
        'classes': ["augment1", "augment2", "augment3"]  # Replace with your classes
    }
    
    pipeline = DomainAdaptationPipeline(
        synthetic_data_config=synthetic_config,
        real_data_dir="test_images",
        output_dir="domain_adapted_model",
        initial_model_path=None  # Will train from scratch
    )
    
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()