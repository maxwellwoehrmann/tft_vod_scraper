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
