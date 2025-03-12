import os
import json
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from tqdm import tqdm
import shutil

def setup_directories(base_dir="augment_dataset"):
    """Create necessary output directories"""
    dirs = {
        "train": os.path.join(base_dir, "train"),
        "val": os.path.join(base_dir, "val")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def process_augment(augment_path, size=(30, 30)):
    """Process a single augment image, optimized for TFT-like display"""
    with Image.open(augment_path) as img:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Resize with high quality
        img = img.resize(size, Image.LANCZOS)
        
        # Split channels for processing
        r, g, b, a = img.split()
        
        # Threshold alpha to make transparency more decisive
        threshold = 25
        a = a.point(lambda x: 0 if x < threshold else x)
        
        # Slightly reduce contrast to minimize bloom
        enhancer = ImageEnhance.Contrast(Image.merge('RGBA', (r, g, b, a)))
        img = enhancer.enhance(0.9)
        
        # Very slight blur to reduce artifacts
        img = img.filter(ImageFilter.GaussianBlur(0.3))
        
        return img

def create_black_background(size=(30, 30)):
    """Create a realistic TFT black background with subtle variations"""
    # Create base black region
    background = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Apply subtle blue-tinted variation to simulate TFT black
    darkness = random.uniform(0.7, 0.9)  # 70-90% darkness
    max_value = int(255 * (1 - darkness))
    
    for y in range(size[1]):
        for x in range(size[0]):
            # Blue-tinted variation (TFT-like)
            r = random.randint(5, max_value - 5)
            g = random.randint(5, max_value)
            b = random.randint(5, int(max_value * 1.2))  # Slightly more blue
            
            # Clamp values
            r, g, b = min(r, 255), min(g, 255), min(b, 255)
            background[y, x] = [b, g, r]  # BGR format
    
    return Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))

def apply_subtle_effects(img):
    """Apply subtle, realistic effects to image"""
    # Convert to numpy for processing
    img_np = np.array(img)
    
    # Randomly apply subtle effects
    effects = []

    # 0. image stretching (10% chance)
    if random.random() < 0.1:
        height, width = img_np.shape[:2]
        
        # Choose stretching type
        stretch_type = random.choice([
            "horizontal", "vertical"
        ])
        
        if stretch_type == "horizontal":
            # Horizontal stretch
            scale_x = random.uniform(0.85, 1.25)
            scale_y = 1.0
            stretch_matrix = np.float32([
                [scale_x, 0, 0],
                [0, scale_y, 0]
            ])
            img_np = cv2.warpAffine(img_np, stretch_matrix, (width, height), 
                                   borderMode=cv2.BORDER_REPLICATE) 
        elif stretch_type == "vertical":
            # Vertical stretch
            scale_x = 1.0
            scale_y = random.uniform(0.85, 1.25)
            stretch_matrix = np.float32([
                [scale_x, 0, 0],
                [0, scale_y, 0]
            ])
            img_np = cv2.warpAffine(img_np, stretch_matrix, (width, height), 
                                   borderMode=cv2.BORDER_REPLICATE)

    # 1. Slight brightness variation (30% chance)
    if random.random() < 0.3:
        brightness = random.uniform(0.9, 1.1)
        img_np = cv2.convertScaleAbs(img_np, alpha=brightness, beta=0)
        effects.append("brightness")
    
    # 2. Subtle noise
    if random.random() < 0.3:
        noise_amount = random.uniform(0.3, 1.2)
        noise = np.random.normal(0, noise_amount, img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)
        effects.append("noise")
    
    # 3. Motion blur
    if random.random() < 0.1:
        kernel_size = random.choice([2, 3])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1/kernel_size  # Horizontal blur
        img_np = cv2.filter2D(img_np, -1, kernel)
        effects.append("motion_blur")
    
    # 4. Compression artifacts (30% chance, mild quality)
    if random.random() < 0.2 and "compression" not in effects:
        is_success, buffer = cv2.imencode(".jpg", img_np, [cv2.IMWRITE_JPEG_QUALITY, random.randint(90, 98)])
        if is_success:
            img_np = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            effects.append("compression")
    
    return Image.fromarray(img_np)

def create_augment_with_background(augment_path, augment_size=(30, 30)):
    """Create a single augment on realistic black background"""

    # Create the black background
    background = create_black_background(augment_size)
    background = background.convert('RGBA')  # Convert to RGBA for transparency
    
    # Process augment
    augment = process_augment(augment_path, augment_size)

    offset_x = 0
    offset_y = 0
    # offset augment
    if random.random() < 0.1:
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)
    
    
    # Paste onto background
    background.paste(augment, (offset_x, offset_y), augment)
    
    # Convert back to RGB for further processing
    result = background.convert('RGB')
    
    # Apply subtle effects
    result = apply_subtle_effects(result)
    
    return result

def extract_augment_class(file_path):
    """Extract augment class identifier from filename"""
    # Get filename without extension and directory path
    filename = os.path.basename(file_path)
    name = os.path.splitext(filename)[0]
    
    # Clean up the name to use as class identifier
    # Remove any invalid characters for directory names
    import re
    clean_name = re.sub(r'[^\w\-]', '_', name)
    
    return clean_name

def generate_dataset(
    samples_per_augment=500,
    output_dir="augment_dataset",
    augments_dir="assets/augments",
    augment_size=(30, 30),
    val_split=0.2
):
    """Generate the complete augment identification dataset"""
    print(f"Setting up directories for augment dataset...")
    dirs = setup_directories(output_dir)
    
    # Get all augment files
    augment_files = []
    for file in os.listdir(augments_dir+"/silver"): #silver folder
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            file_path = os.path.join(augments_dir, file)
            augment_files.append(file_path)

    for file in os.listdir(augments_dir+"/gold"): #gold folder
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            file_path = os.path.join(augments_dir, file)
            augment_files.append(file_path)
    
    for file in os.listdir(augments_dir+"/prismatic"): #prismatic folder
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            file_path = os.path.join(augments_dir, file)
            augment_files.append(file_path)
    
    if not augment_files:
        raise ValueError(f"No augment files found in {augments_dir}")
    
    print(f"Found {len(augment_files)} unique augment files")
    
    # Get augment class names
    augment_classes = {}  # Map augment paths to class names
    for file_path in augment_files:
        augment_class = extract_augment_class(file_path)
        augment_classes[file_path] = augment_class
    
    # Create class directories
    unique_classes = set(augment_classes.values())
    for split in ["train", "val"]:
        for class_name in unique_classes:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    print(f"Creating dataset with {len(unique_classes)} unique augment classes")
    
    # Generate samples for each augment
    total_images = len(augment_files) * samples_per_augment
    print(f"Generating {total_images} training images ({samples_per_augment} per augment type)...")
    
    # Initialize counters for each class and split
    counters = {split: {cls: 0 for cls in unique_classes} for split in ["train", "val"]}
    
    # Track metadata
    metadata = {
        "classes": list(unique_classes),
        "samples_per_augment": samples_per_augment,
        "total_augments": len(augment_files),
        "samples_per_class": {}
    }
    
    # Track class mapping (class index to name)
    class_mapping = {i: cls for i, cls in enumerate(sorted(unique_classes))}
    
    # Use tqdm for progress tracking
    pbar = tqdm(total=total_images)
    
    for augment_path in augment_files:
        augment_class = augment_classes[augment_path]
        
        metadata["samples_per_class"][augment_class] = metadata["samples_per_class"].get(augment_class, 0) + samples_per_augment
        
        for i in range(samples_per_augment):
            # Determine split (train or val)
            split = "val" if random.random() < val_split else "train"
            
            # Create the augment image with background
            augment_img = create_augment_with_background(augment_path, augment_size)
            
            # Save the image
            output_path = os.path.join(
                output_dir, 
                split, 
                augment_class, 
                f"{augment_class}_{counters[split][augment_class]:05d}.jpg"
            )
            augment_img.save(output_path)
            
            # Increment counter
            counters[split][augment_class] += 1
            pbar.update(1)
    
    pbar.close()
    
    # Save metadata
    metadata["samples"] = {
        "train": {cls: counters["train"][cls] for cls in unique_classes},
        "val": {cls: counters["val"][cls] for cls in unique_classes},
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save class mapping
    with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    # Print summary
    print("\nDataset generation complete:")
    print(f"  Total classes: {len(unique_classes)}")
    total_train = sum(counters["train"].values())
    total_val = sum(counters["val"].values())
    print(f"  Total images: {total_train + total_val} ({total_train} training, {total_val} validation)")
    
    print(f"\nDataset saved to {output_dir}")

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "samples_per_augment": 500,  # 100 images per augment file
        "output_dir": "training_dataset/augment_dataset",
        "augments_dir": "assets/augments",
        "augment_size": (30, 30),
        "val_split": 0.2  # 20% validation
    }
    
    # Generate dataset
    generate_dataset(**CONFIG)