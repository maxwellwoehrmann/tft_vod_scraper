import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pathlib import Path
import easyocr
import shutil
from difflib import SequenceMatcher
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO
import torchvision.transforms as transforms
import torch.nn.functional as F

class DebugManager:
    """Manage debug image saving for TFT pipeline diagnostics"""
    
    def __init__(self, debug_enabled=False, base_dir="debug"):
        """
        Initialize debug manager
        
        Args:
            debug_enabled: Whether debug mode is enabled
            base_dir: Base directory for debug output
        """
        self.debug_enabled = debug_enabled
        self.base_dir = base_dir
        self.readers = {}
        self.current_vod_dir = None
        
        if debug_enabled:
            os.makedirs(base_dir, exist_ok=True)
    
    def set_current_vod(self, vod_id):
        """Set current VOD being processed"""
        if not self.debug_enabled:
            return
            
        self.current_vod_dir = os.path.join(self.base_dir, f"vod_{vod_id}")
        os.makedirs(self.current_vod_dir, exist_ok=True)
    
    def get_reader(self, langs):
        """Get or create an EasyOCR reader for the specified languages"""
        if not self.debug_enabled:
            return None
            
        key = '-'.join(sorted(langs))
        if key not in self.readers:
            self.readers[key] = easyocr.Reader(langs)
        return self.readers[key]
    
    def create_frame_folder(self, player, frame_number, timestamp=None):
        """Create folder for a specific frame"""
        if not self.debug_enabled or not self.current_vod_dir:
            return None
            
        folder_name = f"{player}_frame_{frame_number}"
        if timestamp:
            folder_name += f"_t{timestamp:.1f}"
            
        frame_dir = os.path.join(self.current_vod_dir, folder_name)
        os.makedirs(frame_dir, exist_ok=True)
        return frame_dir
    
    def save_scouting_frame(self, frame, frame_dir, suffix="original"):
        """Save a scouting frame"""
        if not self.debug_enabled or not frame_dir:
            return
            
        output_path = os.path.join(frame_dir, f"{suffix}.jpg")
        cv2.imwrite(output_path, frame)
    
    def save_ocr_debug(self, image, players, frame_dir, name_y=None, extended=False):
        """
        Save OCR debug visualization similar to test_ocr.py
        
        Args:
            image: Input image
            players: List of player names to match against
            frame_dir: Debug directory for this frame
            name_y: Y-coordinate of name (for ROI visualization)
            extended: Whether this is using the extended ROI
        """
        if not self.debug_enabled or not frame_dir:
            return
            
        # Define which image we're processing (default or extended)
        suffix = "extended_ocr" if extended else "ocr"
        
        # Create preprocessing versions
        preprocessing_methods = self._preprocess_image(image)
        
        # Run OCR with different language combinations
        lang_combinations = [
            ['en'],
            ['en', 'ch_sim'],
            ['en', 'ja'],
        ]
        
        ocr_results = {}
        
        for langs in lang_combinations:
            reader = self.get_reader(langs)
            lang_key = '_'.join(langs)
            ocr_results[lang_key] = {}
            
            for method_name, processed_img in preprocessing_methods.items():
                try:
                    # Skip RGB for OCR
                    if method_name == 'original':
                        ocr_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
                    else:
                        ocr_img = processed_img
                    
                    ocr_result = reader.readtext(ocr_img)
                    
                    # Store results
                    texts = []
                    for detection in ocr_result:
                        text = detection[1]
                        confidence = detection[2]
                        texts.append({
                            'text': text,
                            'confidence': confidence,
                            'box': detection[0]
                        })
                    
                    ocr_results[lang_key][method_name] = texts
                    
                    # Try to match with known players
                    best_match = None
                    best_ratio = 0
                    
                    for text_obj in texts:
                        text = text_obj['text'].lower()
                        for player in players:
                            ratio = SequenceMatcher(None, text, player.lower()).ratio()
                            if ratio > best_ratio and ratio > 0.6:  # Minimum threshold
                                best_ratio = ratio
                                best_match = {
                                    'player': player,
                                    'text': text,
                                    'confidence': text_obj['confidence'],
                                    'match_ratio': ratio
                                }
                    
                    if best_match:
                        ocr_results[lang_key][method_name + '_best_match'] = best_match
                    
                except Exception as e:
                    ocr_results[lang_key][method_name] = [{'error': str(e)}]
        
        # Create visualization
        self._visualize_ocr_results(image, ocr_results, preprocessing_methods, 
                               os.path.join(frame_dir, f"{suffix}.jpg"), name_y)
        
        # Save OCR results as text file
        self._save_ocr_text_results(ocr_results, os.path.join(frame_dir, f"{suffix}_results.txt"))
    
    def _save_ocr_text_results(self, results, output_path):
        """
        Save OCR results to a text file for easier inspection
        
        Args:
            results: OCR results dictionary
            output_path: Path to save the text file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("OCR Detection Results\n")
                f.write("====================\n\n")
                
                # Extract and sort best matches
                best_matches = []
                for lang_key, lang_results in results.items():
                    for method_name, texts in lang_results.items():
                        if '_best_match' in method_name:
                            match = texts  # This is a single object, not a list
                            best_matches.append({
                                'language': lang_key,
                                'method': method_name.replace('_best_match', ''),
                                'player': match['player'],
                                'text': match['text'],
                                'confidence': match['confidence'],
                                'match_ratio': match['match_ratio']
                            })
                
                # Sort best matches by ratio
                best_matches.sort(key=lambda x: x['match_ratio'], reverse=True)
                
                # Write best matches
                if best_matches:
                    f.write("Best Player Matches:\n")
                    f.write("-------------------\n")
                    for match in best_matches:
                        f.write(f"Language: {match['language']}, Method: {match['method']}\n")
                        f.write(f"Player: {match['player']}\n")
                        f.write(f"OCR Text: '{match['text']}'\n")
                        f.write(f"OCR Confidence: {match['confidence']:.4f}\n")
                        f.write(f"Match Ratio: {match['match_ratio']:.4f}\n\n")
                else:
                    f.write("No good player matches found.\n\n")
                
                # Write all OCR results
                f.write("\nAll OCR Results:\n")
                f.write("--------------\n")
                for lang_key, lang_results in results.items():
                    f.write(f"\nLanguage: {lang_key}\n")
                    for method_name, texts in lang_results.items():
                        if not '_best_match' in method_name:
                            f.write(f"\n  Method: {method_name}\n")
                            if isinstance(texts, list):
                                for i, text_obj in enumerate(texts):
                                    if 'text' in text_obj:
                                        f.write(f"    {i+1}. Text: '{text_obj['text']}'\n")
                                        conf_str = f"{text_obj['confidence']:.4f}" if 'confidence' in text_obj else "N/A"
                                        f.write(f"       Confidence: {conf_str}\n")
                                        if 'box' in text_obj:
                                            f.write(f"       Position: {text_obj['box']}\n")
                                    elif 'error' in text_obj:
                                        f.write(f"    Error: {text_obj['error']}\n")
                            else:
                                f.write(f"    Unexpected result format: {texts}\n")
        except Exception as e:
            print(f"Error saving OCR text results: {e}")

    
    def _preprocess_image(self, img):
        """Apply various preprocessing techniques to an image"""
        # Convert BGR to RGB for display
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Thresholding options
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary_inv_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        adaptive_thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY_INV, 11, 2)
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(adaptive_thresh_inv, cv2.MORPH_OPEN, kernel)
        
        # Edge enhancement
        edges = cv2.Canny(gray, 100, 200)
        enhanced = cv2.addWeighted(gray, 1.5, edges, 0.5, 0)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        
        # Combine methods
        methods = {
            'original': rgb,
            'gray': gray,
            'binary_otsu': binary_otsu,
            'binary_inv_otsu': binary_inv_otsu,
            'adaptive_thresh': adaptive_thresh,
            'adaptive_thresh_inv': adaptive_thresh_inv,
            'opening': opening,
            'edge_enhanced': enhanced,
            'contrast_enhanced': contrast_enhanced
        }
        
        return methods
    
    def _visualize_ocr_results(self, img, results, preprocessing_methods, output_path, name_y=None):
        """Visualize OCR results with different methods"""
        # Count number of methods for layout
        method_count = len(preprocessing_methods)
        lang_count = len(results)
        
        # Create figure for visualization
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"OCR Testing", fontsize=16)
        
        # Plot preprocessing methods
        for i, (method_name, processed_img) in enumerate(preprocessing_methods.items()):
            ax = plt.subplot(2, method_count, i + 1)
            if len(processed_img.shape) == 3:  # Color image
                ax.imshow(processed_img)
            else:  # Grayscale
                ax.imshow(processed_img, cmap='gray')
            ax.set_title(method_name)
            ax.axis('off')
            
            # Add indicator for name position if provided
            if name_y is not None and method_name == 'original':
                # For the small OCR image, we don't want to show the absolute y position
                # Instead, we'll just indicate the row within the image
                ax.axhline(y=processed_img.shape[0]/2, color='r', linestyle='-', linewidth=1)
        
        # Plot a summary of OCR results
        summary_texts = []
        best_matches = []
        
        for lang_key, lang_results in results.items():
            for method_name, texts in lang_results.items():
                if '_best_match' in method_name:
                    match = texts  # This is a single object, not a list
                    best_matches.append(f"{lang_key}+{method_name.replace('_best_match', '')}: "
                                      f"{match['player']} ({match['match_ratio']:.2f})")
                elif isinstance(texts, list):
                    for text_obj in texts:
                        if 'text' in text_obj:
                            conf_str = f"{text_obj['confidence']:.2f}" if 'confidence' in text_obj else "N/A"
                            summary_texts.append(f"{lang_key}+{method_name}: {text_obj['text']} ({conf_str})")
        
        # Create text summary
        ax = plt.subplot(2, 1, 2)
        ax.axis('off')
        
        # Draw best matches first
        if best_matches:
            ax.text(0.01, 0.99, "Best Player Matches:", fontsize=12, weight='bold', 
                   verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
            
            for i, match_text in enumerate(best_matches):
                ax.text(0.01, 0.95 - i*0.04, match_text, fontsize=10, 
                       verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
        
        # Draw all OCR results
        ax.text(0.01, 0.7, "All OCR Results:", fontsize=12, weight='bold', 
               verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
        
        max_results = min(20, len(summary_texts))  # Limit to 20 results
        for i, result_text in enumerate(summary_texts[:max_results]):
            ax.text(0.01, 0.65 - i*0.03, result_text, fontsize=8, 
                   verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
        
        if len(summary_texts) > max_results:
            ax.text(0.01, 0.65 - max_results*0.03, f"... and {len(summary_texts) - max_results} more results", 
                   fontsize=8, style='italic', verticalalignment='top', horizontalalignment='left', 
                   transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
    
    def save_box_detection(self, roi, frame_dir, model_path="runs/box_detection/weights/best.pt"):
        """
        Save box detection visualization
        
        Args:
            roi: Region of interest image
            frame_dir: Debug directory for this frame
            model_path: Path to YOLO model
        """
        if not self.debug_enabled or not frame_dir:
            return
        
        # Load YOLO model
        try:
            model = YOLO(model_path)
            
            # Run detection
            detection_results = model.predict(
                source=roi,
                conf=0.25,
                verbose=False
            )
            
            # Create visualization
            if len(detection_results) > 0:
                result_img = detection_results[0].plot()
                cv2.imwrite(os.path.join(frame_dir, "box_detection.jpg"), result_img)
                
                # If boxes were detected, save ROIs for augment classification
                if len(detection_results[0].boxes) > 0:
                    boxes = detection_results[0].boxes.data.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2, conf, cls = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Extract the box image
                        box_img = roi[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(frame_dir, f"box_{i+1}_conf_{conf:.2f}.jpg"), box_img)
            else:
                # No detection, just save original ROI
                cv2.imwrite(os.path.join(frame_dir, "box_detection.jpg"), roi)
                
        except Exception as e:
            print(f"Error in box detection debug: {e}")
    
    def save_augment_predictions(self, augment_images, frame_dir, 
                                model_path="augment_models/best_model.pth",
                                classes_path="training_dataset/augment_dataset/class_mapping.json",
                                model_type="resnet34"):
        """
        Save augment classification visualizations
        
        Args:
            augment_images: List of augment images
            frame_dir: Debug directory for this frame
            model_path: Path to classification model
            classes_path: Path to class mapping
            model_type: Type of model architecture
        """
        if not self.debug_enabled or not frame_dir:
            return
            
        try:
            # Load model and class mapping
            model, classes, device = self._load_classification_model(
                model_path, classes_path, model_type
            )
            
            # Process each augment image
            for i, img in enumerate(augment_images):
                # Make prediction
                predictions = self._predict_augment(model, img, classes, device, top_k=5)
                
                # Create visualization
                self._visualize_augment_prediction(
                    img, predictions, 
                    os.path.join(frame_dir, f"augment_{i+1}_prediction.jpg")
                )
                
        except Exception as e:
            print(f"Error in augment prediction debug: {e}")
    
    def _load_classification_model(self, model_path, classes_path, model_type="resnet34"):
        """Load classification model for augment prediction"""
        # Load class names
        try:
            with open(classes_path, 'r') as f:
                classes = json.load(f)
            
            # Convert string keys to integers if dictionary
            if isinstance(classes, dict):
                class_map = {}
                for key, value in classes.items():
                    class_map[int(key)] = value
                classes = class_map
        except Exception:
            # Fallback with placeholder classes
            classes = {i: f"class_{i}" for i in range(1000)}
        
        # Set device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # Create model with correct architecture
        if model_type == "resnet18":
            from torchvision import models
            model = models.resnet18(weights=None)
            # Modify first conv layer to match the saved model
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Remove the maxpool layer for small input images
            model.maxpool = torch.nn.Identity()
        elif model_type == "resnet34":
            from torchvision import models
            model = models.resnet34(weights=None)
            # Apply same modifications
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = torch.nn.Identity()
        
        # Determine number of classes from the model weights
        state_dict = torch.load(model_path, map_location='cpu')
        fc_weight_shape = state_dict['fc.weight'].shape
        num_classes = fc_weight_shape[0]
        
        # Adjust final layer to match number of classes
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        # Load the saved state dict
        model.load_state_dict(state_dict)
        
        # Move model to device and set eval mode
        model = model.to(device)
        model.eval()
        
        return model, classes, device
    
    def _predict_augment(self, model, image, classes, device, top_k=5):
        """Predict the class of a single augment image"""
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        image_tensor = transform(image).unsqueeze(0)
        
        # Move to same device as model
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        try:
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            probs, indices = torch.topk(probabilities, top_k)
            probs = probs.squeeze().cpu().numpy()
            indices = indices.squeeze().cpu().numpy()
            
            # Map indices to class names
            predictions = []
            for i, idx in enumerate(indices):
                # Convert numpy int to python int if necessary
                if isinstance(idx, np.integer):
                    idx = idx.item()
                    
                # Get class name safely
                if isinstance(classes, dict):
                    class_name = classes.get(idx, f"unknown_class_{idx}")
                else:
                    class_name = classes[idx] if 0 <= idx < len(classes) else f"unknown_class_{idx}"
                    
                predictions.append({
                    'class': class_name,
                    'probability': float(probs[i])
                })
            
            return predictions
                
        except Exception as e:
            return [{'class': f'error: {str(e)}', 'probability': 0.0}]
    
    def _visualize_augment_prediction(self, img, predictions, output_path):
        """Create visualization of augment prediction"""
        fig = plt.figure(figsize=(8, 6))
        
        # Display image
        if isinstance(img, np.ndarray):
            plt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            plt_img = img
            
        plt.subplot(1, 2, 1)
        plt.imshow(plt_img)
        plt.axis('off')
        plt.title("Augment Image")
        
        # Create bar chart of predictions
        plt.subplot(1, 2, 2)
        
        classes = [p['class'] for p in predictions]
        probs = [p['probability'] for p in predictions]
        
        y_pos = np.arange(len(classes))
        plt.barh(y_pos, probs, align='center')
        plt.yticks(y_pos, classes)
        plt.xlabel('Probability')
        plt.title('Augment Predictions')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)