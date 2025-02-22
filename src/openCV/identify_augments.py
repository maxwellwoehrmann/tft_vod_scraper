import cv2
import numpy as np
from typing import List, Tuple
import os

def classify_augment(image_path):
    """
    Classifies TFT augment images as gold, silver, or prismatic.
    
    Args:
        image_path: Path to the augment image
        
    Returns:
        str: 'gold', 'silver', or 'prismatic'
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    
    # Convert to HSV color space for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create masks for the icon part (excluding background)
    # Blue background removal
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    icon_mask = cv2.bitwise_not(blue_mask)
    
    # Apply mask to get only the icon pixels
    icon = cv2.bitwise_and(img, img, mask=icon_mask)
    hsv_icon = cv2.bitwise_and(hsv, hsv, mask=icon_mask)
    
    # Get non-zero pixels
    non_zero_pixels = hsv_icon[np.where(icon_mask > 0)]
    
    if len(non_zero_pixels) == 0:
        raise ValueError("No valid pixels found after masking")
    
    # Calculate color statistics
    mean_hsv = np.mean(non_zero_pixels, axis=0)
    std_hsv = np.std(non_zero_pixels, axis=0)
    
    # Calculate color ratios
    golden_pixels = np.sum((non_zero_pixels[:, 0] >= 20) & 
                          (non_zero_pixels[:, 0] <= 30) & 
                          (non_zero_pixels[:, 1] >= 150))
    silver_pixels = np.sum((non_zero_pixels[:, 1] < 50) & 
                          (non_zero_pixels[:, 2] > 150))
    prismatic_pixels = np.sum((non_zero_pixels[:, 0] >= 130) & 
                             (non_zero_pixels[:, 0] <= 170) & 
                             (non_zero_pixels[:, 1] >= 50))
    
    total_pixels = len(non_zero_pixels)
    golden_ratio = golden_pixels / total_pixels
    silver_ratio = silver_pixels / total_pixels
    prismatic_ratio = prismatic_pixels / total_pixels
    
    # Classification logic with confidence thresholds
    if golden_ratio > 0.4:
        return 'gold'
    elif prismatic_ratio > 0.15:  # Prismatic often has color variation
        return 'prismatic'
    elif silver_ratio > 0.3:
        return 'silver'
    
    # Fallback to more detailed analysis if initial check is inconclusive
    # Check for specific HSV characteristics
    if mean_hsv[1] > 150 and 20 <= mean_hsv[0] <= 30:  # Strong golden saturation
        return 'gold'
    elif mean_hsv[1] < 50 and mean_hsv[2] > 150:  # Low saturation, high value
        return 'silver'
    elif std_hsv[0] > 20:  # High hue variation indicates prismatic
        return 'prismatic'
    
    # Final fallback based on dominant characteristics
    max_ratio = max(golden_ratio, silver_ratio, prismatic_ratio)
    if max_ratio == golden_ratio:
        return 'gold'
    elif max_ratio == prismatic_ratio:
        return 'prismatic'
    else:
        return 'silver'

    return None

def find_black_regions(image: np.ndarray) -> np.ndarray:
    """
    Find black regions using precise color ranges derived from sampling.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create masks
    # The V channel is our most reliable indicator
    v_mask = cv2.inRange(hsv[:,:,2], 25, 45)
    
    # The S channel can help eliminate some noise
    s_mask = cv2.inRange(hsv[:,:,1], 35, 90)
    
    # RGB ranges
    rgb_mask = cv2.inRange(image, np.array([13, 19, 14]), np.array([45, 37, 41]))
    
    # Combine masks
    combined_mask = cv2.bitwise_and(v_mask, s_mask)
    combined_mask = cv2.bitwise_and(combined_mask, rgb_mask)
    
    # Minor cleanup
    kernel = np.ones((2,2), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    return combined_mask

def find_best_box(mask: np.ndarray, box_width: int = 106, box_height: int = 34) -> Tuple[int, int, int]:
    """
    Finds the optimal position for a box of specified dimensions that covers the maximum number of black pixels.
    
    Args:
        mask: Binary mask where white (255) represents the regions of interest
        box_width: Width of the box to fit (default 90)
        box_height: Height of the box to fit (default 30)
        
    Returns:
        Tuple containing (x, y, pixel_count) where:
            x: x-coordinate of the top-left corner of the best box
            y: y-coordinate of the top-left corner of the best box
            pixel_count: number of white pixels contained in the best box
    """
    if mask.size == 0 or box_width <= 0 or box_height <= 0:
        raise ValueError("Invalid input parameters")
        
    if box_width > mask.shape[1] or box_height > mask.shape[0]:
        raise ValueError("Box dimensions larger than image")
    
    max_pixels = 0
    best_x = 0
    best_y = 0
    
    # Iterate through all possible box positions
    for y in range(mask.shape[0] - box_height + 1):
        for x in range(mask.shape[1] - box_width + 1):
            # Extract the region under the current box
            roi = mask[y:y + box_height, x:x + box_width]
            
            # Count white pixels (255 values) in the region
            pixel_count = np.count_nonzero(roi == 255)
            
            # Update best position if we found more pixels
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                best_x = x
                best_y = y
    
    return best_x, best_y, max_pixels

def detect_augments(image: np.ndarray, roi: Tuple[int, int, int, int] = (1300, 280, 130, 100)):
    """
    Detect and extract augment icons from a TFT screenshot.
    Returns the augment image and number of slots (3 for standard box).
    """
    x, y, w, h = roi
    image_roi = image[y:y+h, x:x+w]
 
    black_mask = find_black_regions(image_roi)
    
    # Find the best box position
    return find_best_box(black_mask)