import os
import json
import random
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass

class TFTTrainingDataGenerator:
    def __init__(
        self,
        augments_dir: str,
        boards_dir: str,
        output_dir: str,
        num_samples: int = 10,
        augment_size: Tuple[int, int] = (30, 30),
        strip_spacing: int = 0,
        board_size: Tuple[int, int] = (130, 100)
    ):
        self.augments_dir = Path(augments_dir)
        self.boards_dir = Path(boards_dir)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.augment_size = augment_size
        self.strip_spacing = strip_spacing
        self.board_size = board_size
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.augment_files = list(self.augments_dir.glob('*.webp'))
        self.board_files = list(self.boards_dir.glob('*.webp'))
        
        self.metadata = {}

    def resize_augment(self, augment_path: Path) -> Image:
        """Resize a single augment to the specified size while preserving original qualities."""
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

    def create_augment_strip(self, selected_augments: List[Path]) -> Image:
        """Create a strip containing 1-3 augment icons with spacing and realistic pseudo-black background."""
        strip_width = (len(selected_augments) * self.augment_size[0] + 
                      (len(selected_augments) - 1) * self.strip_spacing)
        
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
        strip = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        
        # Place each resized augment
        for i, augment_path in enumerate(selected_augments):
            resized_augment = self.resize_augment(augment_path)
            x_pos = i * (self.augment_size[0] + self.strip_spacing)
            strip.paste(resized_augment, (x_pos, 0))
        
        return strip

    def get_random_placement(self, strip_width: int, strip_height: int) -> Tuple[int, int]:
        """Get random coordinates for placing the augment strip within the board."""
        max_x = self.board_size[0] - strip_width
        max_y = self.board_size[1] - strip_height
        return (random.randint(0, max_x), random.randint(0, max_y))

    def generate_sample(self, index: int) -> None:
        """Generate a single training sample with individual augment positions."""
        # Randomly select 1-3 augments and a board
        num_augments = random.randint(1, 3)
        selected_augments = random.sample(self.augment_files, num_augments)
        selected_board = random.choice(self.board_files)
        
        # Create augment strip
        augment_strip = self.create_augment_strip(selected_augments)
        
        # Open and prepare board
        with Image.open(selected_board) as board:
            if board.mode != 'RGB':
                board = board.convert('RGB')
            
            board = board.crop((380, 130, 380 + 60, 130 + 40))
            board = board.resize(self.board_size, Image.Resampling.LANCZOS)
            
            # Get random placement coordinates for whole strip
            x, y = self.get_random_placement(augment_strip.width, augment_strip.height)
            
            # Paste augment strip at random position
            board.paste(augment_strip, (x, y))
            
            # Save the composite image
            output_path = self.output_dir / f'sample_{index:05d}.png'
            board.save(output_path)
        
        # Store metadata with individual augment positions
        augment_positions = []
        for i, augment in enumerate(selected_augments):
            augment_x = x + (i * (self.augment_size[0] + self.strip_spacing))
            augment_positions.append({
                'name': augment.stem,
                'x': augment_x,
                'y': y,
                'width': self.augment_size[0],
                'height': self.augment_size[1]
            })
        
        # This is the key fix - use augment_positions instead of just names
        self.metadata[f'sample_{index:05d}.png'] = {
            'augments': augment_positions,  # Use the full position data
            'panel_origin': {'x': x, 'y': y}
        }

    def generate_dataset(self) -> None:
        """Generate the complete training dataset."""
        print(f"Generating {self.num_samples} training samples...")
        
        for i in range(self.num_samples):
            self.generate_sample(i)
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} samples...")
        
        # Save metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Dataset generation complete. Output saved to {self.output_dir}")

def main():
    # Configuration
    AUGMENTS_DIR = "assets/augments"
    BOARDS_DIR = "assets/boards"
    OUTPUT_DIR = "assets/generated_training"
    NUM_SAMPLES = 15000
    
    # Configure augment size and spacing
    AUGMENT_SIZE = (30, 30)
    STRIP_SPACING = 1
    BOARD_SIZE = (130, 100)
    
    # Create and run generator
    generator = TFTTrainingDataGenerator(
        augments_dir=AUGMENTS_DIR,
        boards_dir=BOARDS_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        augment_size=AUGMENT_SIZE,
        strip_spacing=STRIP_SPACING,
        board_size=BOARD_SIZE
    )
    
    generator.generate_dataset()

if __name__ == "__main__":
    main()