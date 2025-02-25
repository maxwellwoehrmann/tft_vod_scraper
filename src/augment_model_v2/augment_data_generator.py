import os
import json
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import shutil

class AugmentDataGenerator:
    """Generates synthetic training data for augment classification within region crops."""
    
    def __init__(
        self,
        augments_dir: str,
        output_dir: str,
        num_samples: int = 20000,
        augment_size: Tuple[int, int] = (30, 30),
        crop_size: Tuple[int, int] = (130, 100),
        classes_path: str = None
    ):
        """
        Initialize the augment data generator.
        
        Args:
            augments_dir: Directory containing augment images
            output_dir: Output directory for generated data
            num_samples: Number of samples to generate
            augment_size: Size of augment icons
            crop_size: Size of the region crops
            classes_path: Path to file containing class names
        """
        self.augments_dir = Path(augments_dir)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.augment_size = augment_size
        self.crop_size = crop_size
        
        # Create output directories
        self.train_images_dir = self.output_dir / "train" / "images"
        self.train_labels_dir = self.output_dir / "train" / "labels"
        self.val_images_dir = self.output_dir / "val" / "images"
        self.val_labels_dir = self.output_dir / "val" / "labels"
        
        for d in [self.train_images_dir, self.train_labels_dir, 
                 self.val_images_dir, self.val_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load available augments
        self.augment_files = list(self.augments_dir.glob('*.webp'))
        
        # Load classes if provided
        self.classes = []
        if classes_path:
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f]
        else:
            # Extract class names from filenames
            self.classes = sorted(list(set(f.stem for f in self.augment_files)))
        
        # Create mapping from augment name to class index
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        self.metadata = {'train': {}, 'val': {}}
    
    def _generate_black_background(self) -> np.ndarray:
        """
        Generate a realistic black background with TFT-like properties.
        
        Returns:
            Background image as numpy array
        """
        # Create numpy array for the background
        background = np.zeros((self.crop_size[1], self.crop_size[0], 3), dtype=np.uint8)
        
        # Generate random values within the observed ranges for each channel
        for y in range(background.shape[0]):
            for x in range(background.shape[1]):
                # Random RGB values within observed ranges
                r = random.randint(13, 45)
                g = random.randint(19, 37)
                b = random.randint(14, 41)
                background[y, x] = [b, g, r]  # BGR for OpenCV
        
        # Add subtle gradient
        gradient = np.linspace(0.9, 1.1, self.crop_size[0]).reshape(1, self.crop_size[0], 1)
        background = np.clip(background * gradient, 0, 255).astype(np.uint8)
        
        return background
    
    def resize_augment(self, augment_path: Path) -> Tuple[Image.Image, str]:
        """
        Resize a single augment to the specified size with realistic transformations.
        
        Args:
            augment_path: Path to augment image
            
        Returns:
            Tuple of (resized augment image, class name)
        """
        class_name = augment_path.stem
        
        with Image.open(augment_path) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Apply random scaling
            scale_factor = random.uniform(0.9, 1.1)
            scaled_size = (
                int(self.augment_size[0] * scale_factor),
                int(self.augment_size[1] * scale_factor)
            )
            
            # Resize with high-quality settings
            resized = img.resize(scaled_size, Image.Resampling.LANCZOS)
            
            # Apply random transformations
            if random.random() < 0.3:
                # Random rotation
                angle = random.uniform(-5, 5)
                resized = resized.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
            
            # Random brightness/contrast variation
            if random.random() < 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                resized = ImageEnhance.Brightness(resized).enhance(brightness_factor)
            
            if random.random() < 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                resized = ImageEnhance.Contrast(resized).enhance(contrast_factor)
            
            # Create new image with appropriate dimensions
            result = Image.new('RGBA', self.augment_size, (0, 0, 0, 0))
            
            # Center the resized augment
            paste_x = (self.augment_size[0] - resized.width) // 2
            paste_y = (self.augment_size[1] - resized.height) // 2
            
            # Paste resized augment
            result.paste(resized, (paste_x, paste_y), resized)
            
            # Apply blur if desired
            if random.random() < 0.3:
                blur_radius = random.uniform(0.3, 1.0)
                result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Merge with background
            background = Image.new('RGBA', self.augment_size, (0, 0, 0, 255))
            composite = Image.alpha_composite(background, result)
            
            return composite.convert('RGB'), class_name
    
    def generate_sample(self, index: int, phase: str = 'train') -> Dict:
        """
        Generate a single training sample with multiple augments.
        
        Args:
            index: Sample index
            phase: 'train' or 'val'
            
        Returns:
            Metadata for the generated sample
        """
        # Choose number of augments (1-3)
        num_augments = random.choices([1, 2, 3], weights=[0.2, 0.1, 0.7])[0]
        
        # Select random augments
        selected_augments = random.sample(self.augment_files, num_augments)
        
        # Create black background
        background = self._generate_black_background()
        background_pil = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        
        # Determine positions for augments
        width, height = self.crop_size
        spacing = 5  # Space between augments
        
        total_width = num_augments * self.augment_size[0] + (num_augments - 1) * spacing
        start_x = (width - total_width) // 2
        y_pos = (height - self.augment_size[1]) // 2
        
        # Store augment information
        augment_info = []
        
        # Place each augment
        for i, augment_path in enumerate(selected_augments):
            # Resize and transform augment
            augment_img, class_name = self.resize_augment(augment_path)
            
            # Calculate position
            x_pos = start_x + i * (self.augment_size[0] + spacing)
            
            # Paste onto background
            background_pil.paste(augment_img, (x_pos, y_pos))
            
            # Store augment info for label creation
            augment_info.append({
                'class': class_name,
                'class_idx': self.class_to_idx.get(class_name, 0),
                'box': [x_pos, y_pos, x_pos + self.augment_size[0], y_pos + self.augment_size[1]]
            })
        
        # Apply compression artifacts
        if random.random() < 0.5:
            temp_path = self.output_dir / "temp_compression.jpg"
            quality = random.randint(70, 95)
            background_pil.save(temp_path, quality=quality, format='JPEG')
            background_pil = Image.open(temp_path)
            if temp_path.exists():
                os.remove(temp_path)
        
        # Save image
        if phase == 'train':
            image_dir = self.train_images_dir
            label_dir = self.train_labels_dir
        else:
            image_dir = self.val_images_dir
            label_dir = self.val_labels_dir
            
        image_name = f'sample_{index:05d}.png'
        image_path = image_dir / image_name
        background_pil.save(image_path)
        
        # Create YOLO label
        with open(label_dir / f'sample_{index:05d}.txt', 'w') as f:
            for augment in augment_info:
                # Convert to YOLO format (x_center, y_center, width, height) normalized
                x1, y1, x2, y2 = augment['box']
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                f.write(f"{augment['class_idx']} {x_center} {y_center} {box_width} {box_height}\n")
        
        # Store metadata
        metadata = {
            'augments': [{'class': a['class'], 'box': a['box']} for a in augment_info],
            'num_augments': num_augments
        }
        
        return metadata
    
    def generate_dataset(self, train_ratio: float = 0.8) -> None:
        """
        Generate the complete training dataset.
        
        Args:
            train_ratio: Ratio of samples to use for training
        """
        print(f"Generating {self.num_samples} training samples for augment classification...")
        
        # Determine split
        train_samples = int(self.num_samples * train_ratio)
        val_samples = self.num_samples - train_samples
        
        # Generate training samples
        for i in range(train_samples):
            self.metadata['train'][f'sample_{i:05d}.png'] = self.generate_sample(i, phase='train')
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{train_samples} training samples...")
        
        # Generate validation samples
        for i in range(val_samples):
            self.metadata['val'][f'sample_{i:05d}.png'] = self.generate_sample(i, phase='val')
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{val_samples} validation samples...")
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Create dataset.yaml for YOLO training
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.classes),
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset generation complete. Output saved to {self.output_dir}")