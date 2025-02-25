import os
import json
import random
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass
from io import BytesIO

class EnhancedImageEffects:
    @staticmethod
    def add_realistic_noise(image: Image.Image, variance_range=(0.001, 0.02)) -> Image.Image:
        """Add more realistic sensor noise with random variance."""
        img_arr = np.array(image).astype(np.float32)
        
        # Add gaussian noise
        variance = random.uniform(*variance_range)
        gaussian = np.random.normal(0, np.sqrt(variance), img_arr.shape) * 255
        
        # Add poisson noise (simulates sensor noise)
        poisson = np.random.poisson(img_arr).astype(np.float32) - img_arr
        
        # Combine noise types
        noisy_img = img_arr + gaussian * 0.7 + poisson * 0.3
        return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))

    @staticmethod
    def add_compression_artifacts(image: Image.Image, quality_range=(65, 85)) -> Image.Image:
        """Simulate JPEG compression artifacts."""
        # Save with random JPEG quality
        quality = random.randint(*quality_range)
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        return Image.open(buffer)

    @staticmethod
    def add_realistic_blur(image: Image.Image) -> Image.Image:
        """Add realistic camera blur effects."""
        img_arr = np.array(image)
        
        # Random selection of blur types
        blur_type = random.choice(['gaussian', 'motion', 'defocus'])
        
        if blur_type == 'gaussian':
            sigma = random.uniform(0.1, 0.5)
            return image.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        elif blur_type == 'motion':
            # Simulate subtle motion blur
            kernel_size = random.randint(2, 3)
            angle = random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = 1
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size)) / kernel_size
            return Image.fromarray(cv2.filter2D(img_arr, -1, kernel))
        
        else:  # defocus
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.3)))

    @staticmethod
    def adjust_color_distribution(image: Image.Image) -> Image.Image:
        """Adjust color distribution to match real-world variations."""
        # Convert to HSV for more realistic color manipulation
        img_arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Randomly adjust color temperature
        temp_shift = random.uniform(-10, 10)
        img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] + temp_shift, 0, 179)
        
        # Add subtle color variations
        sat_factor = random.uniform(0.9, 1.1)
        img_arr[:, :, 1] = np.clip(img_arr[:, :, 1] * sat_factor, 0, 255)
        
        # Convert back to RGB
        return Image.fromarray(cv2.cvtColor(img_arr.astype(np.uint8), cv2.COLOR_HSV2RGB))

    @staticmethod
    def simulate_screen_effects(image: Image.Image) -> Image.Image:
        """Simulate realistic screen capture artifacts."""
        # Add subtle moire pattern
        img_arr = np.array(image).astype(np.float32)
        x, y = np.meshgrid(np.linspace(0, 1, img_arr.shape[1]), np.linspace(0, 1, img_arr.shape[0]))
        pattern = np.sin(x * 2 * np.pi * random.randint(100, 200)) * np.sin(y * 2 * np.pi * random.randint(100, 200))
        pattern = pattern.reshape(*pattern.shape, 1) * 0.02  # Very subtle effect
        
        img_arr = img_arr * (1 + pattern)
        return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))

    @classmethod
    def apply_random_effects(cls, image: Image.Image) -> Image.Image:
        """Apply a random combination of realistic effects."""
        effects = [
            (cls.add_realistic_noise, 0.9),
            (cls.add_compression_artifacts, 0.7),
            (cls.add_realistic_blur, 0.6),
            (cls.adjust_color_distribution, 0.8),
            (cls.simulate_screen_effects, 0.5)
        ]
        
        result = image
        for effect, probability in effects:
            if random.random() < probability:
                result = effect(result)
        
        return result

class ImageNormalizer:
    @staticmethod
    def normalize_image(image: Image.Image) -> Image.Image:
        """Normalize image brightness and contrast."""
        # Convert to numpy array
        img_arr = np.array(image)
        
        # Convert to float32 for processing
        img_float = img_arr.astype(np.float32)
        
        # Normalize to 0-1 range
        img_norm = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert back to uint8
        return Image.fromarray(img_norm.astype(np.uint8))

class TFTTrainingDataGenerator:
    def __init__(
        self,
        augments_dir: str,
        boards_dir: str,
        output_dir: str,
        num_samples: int = 10,
        augment_size: Tuple[int, int] = (35, 35),  # Increased base size
        strip_spacing: int = 0,
        board_size: Tuple[int, int] = (130, 100),
        size_variance: float = 0.15  # New parameter for size variation
    ):
        self.augments_dir = Path(augments_dir)
        self.boards_dir = Path(boards_dir)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.augment_size = augment_size
        self.strip_spacing = strip_spacing
        self.board_size = board_size
        self.size_variance = size_variance
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.augment_files = list(self.augments_dir.glob('*.webp'))
        self.board_files = list(self.boards_dir.glob('*.webp'))
        
        self.metadata = {}

    def resize_augment(self, augment_path: Path) -> Tuple[Image.Image, Tuple[int, int]]:
        """Resize a single augment without background. Returns the processed image and its size."""
        with Image.open(augment_path) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Apply random size variation
            size_factor = 1.0 + random.uniform(-self.size_variance, self.size_variance)
            varied_size = (
                int(self.augment_size[0] * size_factor),
                int(self.augment_size[1] * size_factor)
            )
            
            # Resize with high-quality settings
            resized = img.resize(varied_size, Image.Resampling.LANCZOS)
            
            # Split into RGBA channels
            r, g, b, a = resized.split()
            
            # Threshold the alpha channel
            threshold = 30
            a = a.point(lambda x: 0 if x < threshold else x)
            
            # Merge channels back
            resized = Image.merge('RGBA', (r, g, b, a))
            
            return resized, varied_size

    def create_augment_strip(self, selected_augments: List[Path]) -> Tuple[Image.Image, List[Tuple[int, int]]]:
        """Create a strip of augment icons with effects. Returns the strip and list of augment sizes."""
        # Process all augments first to get their sizes
        processed_augments = [self.resize_augment(path) for path in selected_augments]
        augment_sizes = [size for _, size in processed_augments]
        
        # Calculate total width including spacing
        total_width = sum(size[0] for size in augment_sizes) + (len(selected_augments) - 1) * self.strip_spacing
        max_height = max(size[1] for size in augment_sizes)
        
        # Create single consistent background with blue-tinted dark values
        background = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        for y in range(background.shape[0]):
            for x in range(background.shape[1]):
                # Adjusted ranges for blue-tinted background
                r = random.randint(10, 20)  # Less red
                g = random.randint(15, 25)  # Slightly more green
                b = random.randint(25, 35)  # More blue
                background[y, x] = [b, g, r]  # BGR for OpenCV
        
        strip = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        
        # Place each processed augment onto the strip
        current_x = 0
        for img, size in processed_augments:
            # Center vertically if shorter than max height
            y_offset = (max_height - size[1]) // 2
            strip.paste(img, (current_x, y_offset), img)  # Use alpha channel for pasting
            current_x += size[0] + self.strip_spacing
        
        strip = EnhancedImageEffects.apply_random_effects(strip)
        
        return strip, augment_sizes

    def get_random_placement(self, strip_width: int, strip_height: int) -> Tuple[int, int]:
        """Get random coordinates for placing the augment strip."""
        max_x = self.board_size[0] - strip_width
        max_y = self.board_size[1] - strip_height
        return (random.randint(0, max_x), random.randint(0, max_y))

    def generate_sample(self, index: int) -> None:
        """Generate a single training sample with effects."""
        num_augments = random.randint(1, 3)
        selected_augments = random.sample(self.augment_files, num_augments)
        selected_board = random.choice(self.board_files)
        
        # Create augment strip and get sizes
        augment_strip, augment_sizes = self.create_augment_strip(selected_augments)
        
        # Process board
        with Image.open(selected_board) as board:
            if board.mode != 'RGB':
                board = board.convert('RGB')
            
            board = board.crop((380, 130, 380 + 60, 130 + 40))
            board = board.resize(self.board_size, Image.Resampling.LANCZOS)
            
            # Get placement coordinates
            x, y = self.get_random_placement(augment_strip.width, augment_strip.height)
            
            # Paste augment strip
            board.paste(augment_strip, (x, y))
            
            # Normalize the entire composite image
            final_image = ImageNormalizer.normalize_image(board)
            
            # Save the final image
            output_path = self.output_dir / f'sample_{index:05d}.png'
            final_image.save(output_path)
        
        # Store metadata with individual augment positions and sizes
        augment_positions = []
        current_x = x
        for i, (augment, size) in enumerate(zip(selected_augments, augment_sizes)):
            y_pos = y + (augment_strip.height - size[1]) // 2  # Account for vertical centering
            augment_positions.append({
                'name': augment.stem,
                'x': current_x,
                'y': y_pos,
                'width': size[0],
                'height': size[1]
            })
            current_x += size[0] + self.strip_spacing
        
        self.metadata[f'sample_{index:05d}.png'] = {
            'augments': augment_positions,
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


if __name__ == "__main__":
    # Configuration
    AUGMENTS_DIR = "assets/augments"
    BOARDS_DIR = "assets/boards"
    OUTPUT_DIR = "assets/generated_training"
    NUM_SAMPLES = 20000
    
    # Configure augment size and spacing
    AUGMENT_SIZE = (30, 30)  # Increased base size
    STRIP_SPACING = 1
    BOARD_SIZE = (130, 100)
    SIZE_VARIANCE = 0.15  # 15% size variation
    
    # Create and run generator
    generator = TFTTrainingDataGenerator(
        augments_dir=AUGMENTS_DIR,
        boards_dir=BOARDS_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        augment_size=AUGMENT_SIZE,
        strip_spacing=STRIP_SPACING,
        board_size=BOARD_SIZE,
        size_variance=SIZE_VARIANCE
    )
    
    generator.generate_dataset()