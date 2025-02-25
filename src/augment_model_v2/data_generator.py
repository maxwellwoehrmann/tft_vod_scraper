import os
import json
import random
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import shutil

class RegionDataGenerator:
    """Generates synthetic training data for augment region detection within fixed ROI."""
    
    def __init__(
        self,
        augments_dir: str,
        boards_dir: str,
        output_dir: str,
        num_samples: int = 10000,
        augment_size: Tuple[int, int] = (30, 30),
        strip_spacing: int = 1,
        roi_size: Tuple[int, int] = (130, 100),
        board_crop_coords: Tuple[int, int, int, int] = (380, 130, 380 + 60, 130 + 40)
    ):
        """
        Initialize the region data generator.
        
        Args:
            augments_dir: Directory containing augment images
            boards_dir: Directory containing board images
            output_dir: Output directory for generated data
            num_samples: Number of samples to generate
            augment_size: Size of augment icons
            strip_spacing: Spacing between augments in strip
            roi_size: Size of the region of interest (width, height)
            board_crop_coords: Coordinates to crop from board images (x1, y1, x2, y2)
        """
        self.augments_dir = Path(augments_dir)
        self.boards_dir = Path(boards_dir)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.augment_size = augment_size
        self.strip_spacing = strip_spacing
        self.roi_size = roi_size
        self.board_crop_coords = board_crop_coords
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Load available augments and boards
        self.augment_files = list(self.augments_dir.glob('*.webp'))
        self.board_files = list(self.boards_dir.glob('*.webp'))
        
        self.metadata = {}
    
    def _generate_black_region(
        self, 
        num_augments: int
    ) -> Tuple[np.ndarray, int, int]:
        """
        Generate a realistic black region with TFT-like properties.
        
        Args:
            num_augments: Number of augments (determines width)
            
        Returns:
            Tuple of (region image, width, height)
        """
        # Calculate strip width
        strip_width = (num_augments * self.augment_size[0] + 
                      (num_augments - 1) * self.strip_spacing)
        
        # Create numpy array for the background with the proper shape
        background = np.zeros((self.augment_size[1], strip_width, 3), dtype=np.uint8)
        
        # Generate random values within the observed ranges for each channel
        for y in range(background.shape[0]):
            for x in range(background.shape[1]):
                # Random RGB values within observed ranges
                r = random.randint(13, 45)
                g = random.randint(19, 37)
                b = random.randint(14, 41)
                background[y, x] = [b, g, r]  # BGR for OpenCV
        
        # Convert numpy array to PIL Image
        strip = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        
        return strip, strip_width, self.augment_size[1]
    
    def resize_augment(self, augment_path: Path) -> Image.Image:
        """
        Resize a single augment to the specified size while preserving original qualities.
        
        Args:
            augment_path: Path to augment image
            
        Returns:
            Resized augment image
        """
        with Image.open(augment_path) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Resize with high-quality settings
            resized = img.resize(self.augment_size, Image.Resampling.LANCZOS)
            
            # Split into RGBA channels
            r, g, b, a = resized.split()
            
            # Threshold the alpha channel to be more decisive about what's background
            # This preserves the glare while still removing the background
            threshold = 30  # Adjust this value to fine-tune the effect
            a = a.point(lambda x: 0 if x < threshold else x)
            
            # Merge channels back
            resized = Image.merge('RGBA', (r, g, b, a))
            
            # Create background and composite
            background = Image.new('RGBA', self.augment_size, (0, 0, 0, 255))
            composite = Image.alpha_composite(background, resized)
            
            return composite.convert('RGB')
    
    def create_augment_strip(self, selected_augments: List[Path]) -> Tuple[Image.Image, int, int]:
        """
        Create a strip containing 1-3 augment icons.
        
        Args:
            selected_augments: List of paths to augment images
            
        Returns:
            Tuple of (augment strip image, width, height)
        """
        num_augments = len(selected_augments)
        
        # Create black region with realistic TFT-like black background values
        strip_array, strip_width, strip_height = self._generate_black_region(num_augments)
        strip = Image.fromarray(strip_array)
        
        # Place each resized augment
        for i, augment_path in enumerate(selected_augments):
            resized_augment = self.resize_augment(augment_path)
            x_pos = i * (self.augment_size[0] + self.strip_spacing)
            strip.paste(resized_augment, (x_pos, 0))
        
        # Apply realistic motion blur and noise
        if random.random() < 0.4:  # 40% chance of motion blur
            strip_cv = cv2.cvtColor(np.array(strip), cv2.COLOR_RGB2BGR)
            kernel_size = random.randint(3, 7)
            kernel = np.zeros((kernel_size, kernel_size))
            
            # Horizontal blur is most common
            kernel[kernel_size//2, :] = 1/kernel_size
            
            strip_cv = cv2.filter2D(strip_cv, -1, kernel)
            strip = Image.fromarray(cv2.cvtColor(strip_cv, cv2.COLOR_BGR2RGB))
        
        # Add compression artifacts
        if random.random() < 0.6:  # 60% chance of compression artifacts
            # Save with low quality and reload
            temp_path = self.output_dir / "temp_compression.jpg"
            strip.save(temp_path, quality=random.randint(60, 95))
            strip = Image.open(temp_path)
            
            # Delete temp file
            if temp_path.exists():
                os.remove(temp_path)
        
        return strip, strip_width, strip_height
    
    def generate_sample(self, index: int) -> Dict:
        """
        Generate a single training sample.
        
        Args:
            index: Sample index
            
        Returns:
            Metadata for the generated sample
        """
        # Randomly select 0-3 augments (with bias toward 3)
        num_augments = random.choices([0, 1, 2, 3], weights=[0.1, 0.2, 0.1, 0.6])[0]
        
        # Select a background board
        selected_board = random.choice(self.board_files)
        
        with Image.open(selected_board) as board:
            if board.mode != 'RGB':
                board = board.convert('RGB')
            
            # Use the specific crop coordinates from original implementation
            board = board.crop(self.board_crop_coords)
            board = board.resize(self.roi_size, Image.Resampling.LANCZOS)
            
            # Define output path now
            output_path = self.images_dir / f'sample_{index:05d}.png'
            
            if num_augments == 0:
                # This is a negative sample (no augments)
                board.save(output_path)
                
                # Create empty label file
                with open(self.labels_dir / f'sample_{index:05d}.txt', 'w') as f:
                    pass  # Empty file for no annotations
                
                # Store metadata
                metadata = {
                    'has_augments': False,
                    'background': selected_board.stem
                }
            else:
                # This is a positive sample
                selected_augments = random.sample(self.augment_files, num_augments)
                augment_strip, strip_width, strip_height = self.create_augment_strip(selected_augments)
                
                # Random placement of strip within ROI (with margin constraints)
                max_x = self.roi_size[0] - strip_width
                max_y = self.roi_size[1] - strip_height
                
                if max_x <= 0 or max_y <= 0:
                    # If strip is larger than ROI, center it
                    x_pos = (self.roi_size[0] - strip_width) // 2
                    y_pos = (self.roi_size[1] - strip_height) // 2
                else:
                    # Random position
                    x_pos = random.randint(0, max_x)
                    y_pos = random.randint(0, max_y)
                
                # Paste strip onto board
                board.paste(augment_strip, (x_pos, y_pos))
                board.save(output_path)
                
                # Create YOLO label
                with open(self.labels_dir / f'sample_{index:05d}.txt', 'w') as f:
                    # Format: class_id x_center y_center width height (normalized)
                    x_center = (x_pos + strip_width / 2) / self.roi_size[0]
                    y_center = (y_pos + strip_height / 2) / self.roi_size[1]
                    width_norm = strip_width / self.roi_size[0]
                    height_norm = strip_height / self.roi_size[1]
                    
                    # Class ID 0 for augment region
                    f.write(f"0 {x_center} {y_center} {width_norm} {height_norm}\n")
                
                # Store metadata
                metadata = {
                    'has_augments': True,
                    'num_augments': num_augments,
                    'augments': [a.stem for a in selected_augments],
                    'region': {
                        'x': x_pos,
                        'y': y_pos,
                        'width': strip_width,
                        'height': strip_height
                    },
                    'background': selected_board.stem
                }
        
        return metadata
    
    def generate_dataset(self) -> None:
        """Generate the complete training dataset."""
        print(f"Generating {self.num_samples} training samples for region detection...")
        
        metadata = {}
        for i in range(self.num_samples):
            metadata[f'sample_{i:05d}.png'] = self.generate_sample(i)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} samples...")
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create dataset.yaml for YOLO training
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images',
            'val': 'images',  # We'll split this during training
            'nc': 1,  # One class: augment region
            'names': {0: 'augment_region'}
        }
        
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset generation complete. Output saved to {self.output_dir}")