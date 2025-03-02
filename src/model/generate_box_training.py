import os
import json
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from tqdm import tqdm

def setup_directories(base_dir="training_dataset"):
    """Create necessary output directories"""
    dirs = {
        "images": os.path.join(base_dir, "images"),
        "labels": os.path.join(base_dir, "labels"),
        "train_images": os.path.join(base_dir, "images", "train"),
        "train_labels": os.path.join(base_dir, "labels", "train"),
        "val_images": os.path.join(base_dir, "images", "val"),
        "val_labels": os.path.join(base_dir, "labels", "val")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def load_augments(augments_dir="assets/augments"):
    """Load and categorize augment images by color"""
    augments = []
    
    for file in os.listdir(augments_dir):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue
        
        file_path = os.path.join(augments_dir, file)
        augments.append(file_path)
    return augments

def load_backgrounds(backgrounds_dir="assets/backgrounds"):
    """Load background images"""
    backgrounds = []
    
    for file in os.listdir(backgrounds_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            file_path = os.path.join(backgrounds_dir, file)
            backgrounds.append(file_path)
    
    if not backgrounds:
        raise ValueError(f"No background images found in {backgrounds_dir}")
    
    print(f"Loaded {len(backgrounds)} background images")
    return backgrounds

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

def create_black_strip(num_augments, augment_size=(30, 30), spacing=0):
    """Create a realistic TFT black strip with subtle variations"""
    width = num_augments * augment_size[0] + (num_augments - 1) * spacing
    height = augment_size[1]
    
    # Create base black region
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Apply subtle blue-tinted variation to simulate TFT black
    darkness = random.uniform(0.7, 0.9)  # 70-90% darkness
    max_value = int(255 * (1 - darkness))
    
    for y in range(height):
        for x in range(width):
            # Blue-tinted variation (TFT-like)
            r = random.randint(5, max_value - 5)
            g = random.randint(5, max_value)
            b = random.randint(5, int(max_value * 1.2))  # Slightly more blue
            
            # Clamp values
            r, g, b = min(r, 255), min(g, 255), min(b, 255)
            strip[y, x] = [b, g, r]  # BGR format
    
    return Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))

def apply_subtle_effects(img):
    """Apply subtle, realistic effects to image"""
    # Convert to numpy for processing
    img_np = np.array(img)
    
    # Randomly apply subtle effects
    effects = []
    
    # 1. Slight brightness variation (30% chance)
    if random.random() < 0.3:
        brightness = random.uniform(0.9, 1.1)
        img_np = cv2.convertScaleAbs(img_np, alpha=brightness, beta=0)
        effects.append("brightness")
    
    # 2. Subtle noise (60% chance, very low amount)
    if random.random() < 0.6:
        noise_amount = random.uniform(0.5, 2)
        noise = np.random.normal(0, noise_amount, img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)
        effects.append("noise")
    
    # 3. Motion blur (20% chance, very slight)
    if random.random() < 0.2:
        kernel_size = random.choice([2, 3])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1/kernel_size  # Horizontal blur
        img_np = cv2.filter2D(img_np, -1, kernel)
        effects.append("motion_blur")
    
    # 4. Compression artifacts (40% chance, mild quality)
    if random.random() < 0.4 and "compression" not in effects:
        is_success, buffer = cv2.imencode(".jpg", img_np, [cv2.IMWRITE_JPEG_QUALITY, random.randint(85, 95)])
        if is_success:
            img_np = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            effects.append("compression")
    
    return Image.fromarray(img_np)

def create_augment_strip(num_augments, augments_dict, augment_size=(30, 30), spacing=0):
    """Create a black strip with augments at fixed positions"""
    # Create the black background strip
    strip = create_black_strip(num_augments, augment_size, spacing)
    strip = strip.convert('RGBA')  # Convert to RGBA for transparency
    
    # Add augments at fixed positions
    for i in range(num_augments):
        augment_path = random.choice(augments_dict)
        
        # Process augment
        augment = process_augment(augment_path, augment_size)
        
        # Paste at fixed position
        x_pos = i * (augment_size[0] + spacing)
        strip.paste(augment, (x_pos, 0), augment)
    
    # Convert back to RGB for further processing
    strip = strip.convert('RGB')
    
    return strip

def generate_yolo_label(box_pos, box_size, image_size):
    """Generate YOLO format label"""
    x, y = box_pos
    w, h = box_size
    img_w, img_h = image_size
    
    # Calculate normalized center coordinates and dimensions
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    norm_width = w / img_w
    norm_height = h / img_h
    
    return f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"

def generate_dataset(
    num_images=15000,
    output_dir="training_dataset/box_dataset",
    augments_dir="assets/augments",
    backgrounds_dir="assets/backgrounds",
    box_sizes=[(30, 30), (60, 30), (90, 30)],
    box_probabilities=[0.1, 0.15, 0.75],
    image_size=160,
    crop_size=80,
    no_box_probability=0.05
):
    """Generate the complete dataset"""
    dirs = setup_directories(output_dir)
    augments = load_augments(augments_dir)
    backgrounds = load_backgrounds(backgrounds_dir)
    
    metadata = {}
    
    print(f"Generating {num_images} training images...")
    for i in tqdm(range(num_images)):
        # Select background
        bg_path = random.choice(backgrounds)
        
        with Image.open(bg_path) as bg:
            if bg.mode != 'RGB':
                bg = bg.convert('RGB')
            
            # Random crop (80x80) and resize to 160x160
            crop_x = random.randint(0, max(0, bg.width - crop_size))
            crop_y = random.randint(0, max(0, bg.height - crop_size))
            cropped = bg.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
            resized_bg = cropped.resize((image_size, image_size), Image.LANCZOS)
            
            # Decide whether to include a box
            include_box = random.random() > no_box_probability
            
            if include_box:
                # Choose box size based on probabilities
                box_size_idx = random.choices(range(len(box_sizes)), box_probabilities)[0]
                box_size = box_sizes[box_size_idx]
                
                # Number of augments based on box width
                num_augments = box_size[0] // 30
                
                # Create augment strip
                strip = create_augment_strip(num_augments, augments)
                
                # Random position for strip
                max_x = image_size - strip.width
                max_y = image_size - strip.height
                box_x = random.randint(0, max_x)
                box_y = random.randint(0, max_y)
                
                # Paste strip onto background
                resized_bg.paste(strip, (box_x, box_y))
                
                # Apply subtle effects to final image
                final_img = apply_subtle_effects(resized_bg)
                
                # Generate label
                label = generate_yolo_label((box_x, box_y), (strip.width, strip.height), (image_size, image_size))
                
                # Store metadata
                metadata[f"{i:05d}"] = {
                    "has_box": True,
                    "box_size": list(box_size),
                    "position": [box_x, box_y],
                    "num_augments": num_augments
                }
                
                # Save label
                with open(os.path.join(dirs["labels"], f"{i:05d}.txt"), "w") as f:
                    f.write(label)
            else:
                # No box, just apply subtle effects
                final_img = apply_subtle_effects(resized_bg)
                
                # Empty label file
                open(os.path.join(dirs["labels"], f"{i:05d}.txt"), "w").close()
                
                # Store metadata
                metadata[f"{i:05d}"] = {
                    "has_box": False
                }
            
            # Save image
            final_img.save(os.path.join(dirs["images"], f"{i:05d}.jpg"))
    
    # Create train/val split
    all_indices = list(range(num_images))
    random.shuffle(all_indices)
    val_size = int(num_images * 0.2)  # 20% validation
    
    train_indices = all_indices[val_size:]
    val_indices = all_indices[:val_size]
    
    # Move files to train/val directories
    for idx in train_indices:
        idx_str = f"{idx:05d}"
        src_img = os.path.join(dirs["images"], f"{idx_str}.jpg")
        dst_img = os.path.join(dirs["train_images"], f"{idx_str}.jpg")
        src_label = os.path.join(dirs["labels"], f"{idx_str}.txt")
        dst_label = os.path.join(dirs["train_labels"], f"{idx_str}.txt")
        
        if os.path.exists(src_img) and os.path.exists(src_label):
            os.rename(src_img, dst_img)
            os.rename(src_label, dst_label)
    
    for idx in val_indices:
        idx_str = f"{idx:05d}"
        src_img = os.path.join(dirs["images"], f"{idx_str}.jpg")
        dst_img = os.path.join(dirs["val_images"], f"{idx_str}.jpg")
        src_label = os.path.join(dirs["labels"], f"{idx_str}.txt")
        dst_label = os.path.join(dirs["val_labels"], f"{idx_str}.txt")
        
        if os.path.exists(src_img) and os.path.exists(src_label):
            os.rename(src_img, dst_img)
            os.rename(src_label, dst_label)
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create data.yaml for YOLO training
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

nc: 1
names: ['box']
""")
    
    print(f"Dataset generated: {len(train_indices)} training and {len(val_indices)} validation images")

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "num_images": 15000,
        "output_dir": "training_dataset/box_dataset",
        "augments_dir": "assets/augments",
        "backgrounds_dir": "assets/boards",
        "box_sizes": [(30, 30), (60, 30), (90, 30)],
        "box_probabilities": [0.1, 0.15, 0.75],  # Focus on 90x30 boxes
        "image_size": 160,
        "crop_size": 80,
        "no_box_probability": 0.05  # 5% images without boxes
    }
    
    # Generate dataset
    generate_dataset(**CONFIG)