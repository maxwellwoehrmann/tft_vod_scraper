# test_ocr.py
import os
import cv2
import numpy as np
import easyocr
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

class OCRTester:
    def __init__(self):
        self.readers = {}
        self.load_test_data()
    
    def load_test_data(self):
        """Load test data from the extraction process"""
        try:
            with open("test_data/players.json", "r") as f:
                self.player_data = json.load(f)
                self.players = self.player_data["players"]
                print(f"Loaded {len(self.players)} players: {', '.join(self.players)}")
        except Exception as e:
            print(f"Error loading player data: {e}")
            self.players = []
    
    def get_reader(self, langs):
        """Get or create an EasyOCR reader for the specified languages"""
        key = '-'.join(sorted(langs))
        if key not in self.readers:
            print(f"Loading OCR reader for languages: {langs}")
            self.readers[key] = easyocr.Reader(langs)
        return self.readers[key]
    
    def preprocess_image(self, img):
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
    
    def run_ocr(self, img, lang_combinations=None):
        """Run OCR with different language combinations and preprocessing methods"""
        if lang_combinations is None:
            lang_combinations = [
                ['en'],
                ['en', 'ch_sim'],
                ['en', 'ja'],
            ]
        
        preprocessing_methods = self.preprocess_image(img)
        
        results = {}
        
        for langs in lang_combinations:
            reader = self.get_reader(langs)
            lang_key = '_'.join(langs)
            results[lang_key] = {}
            
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
                    
                    results[lang_key][method_name] = texts
                    
                    # Try to match with known players
                    best_match = None
                    best_ratio = 0
                    
                    for text_obj in texts:
                        text = text_obj['text'].lower()
                        for player in self.players:
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
                        results[lang_key][method_name + '_best_match'] = best_match
                    
                except Exception as e:
                    results[lang_key][method_name] = [{'error': str(e)}]
        
        return results, preprocessing_methods
    
    def visualize_results(self, img_path, results, preprocessing_methods):
        """Visualize OCR results with different methods"""
        # Count number of methods for layout
        method_count = len(preprocessing_methods)
        lang_count = len(results)
        
        # Create figure for visualization
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"OCR Testing for {os.path.basename(img_path)}", fontsize=16)
        
        # Plot preprocessing methods
        for i, (method_name, processed_img) in enumerate(preprocessing_methods.items()):
            ax = plt.subplot(2, method_count, i + 1)
            if len(processed_img.shape) == 3:  # Color image
                ax.imshow(processed_img)
            else:  # Grayscale
                ax.imshow(processed_img, cmap='gray')
            ax.set_title(method_name)
            ax.axis('off')
        
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
        
        # Save the visualization
        output_dir = os.path.dirname(img_path)
        output_path = os.path.join(output_dir, os.path.basename(img_path).replace('.', '_analysis.'))
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        
        return output_path
    
    def test_image(self, img_path):
        """Test OCR on a single image"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image {img_path}")
            return
        
        print(f"Testing OCR on {img_path}")
        results, preprocessing_methods = self.run_ocr(img)
        
        # Find best match across all methods
        best_match = None
        best_ratio = 0
        
        for lang_key, lang_results in results.items():
            for method_name, data in lang_results.items():
                if '_best_match' in method_name and isinstance(data, dict):
                    if data['match_ratio'] > best_ratio:
                        best_ratio = data['match_ratio']
                        best_match = {
                            'lang': lang_key,
                            'method': method_name.replace('_best_match', ''),
                            'player': data['player'],
                            'text': data['text'],
                            'confidence': data['confidence'],
                            'match_ratio': data['match_ratio']
                        }
        
        if best_match:
            print(f"Best match: {best_match['player']} (match ratio: {best_match['match_ratio']:.2f})")
            print(f"  OCR text: '{best_match['text']}' using {best_match['lang']} + {best_match['method']}")
            print(f"  OCR confidence: {best_match['confidence']:.2f}")
        else:
            print("No good player match found")
        
        # Visualize results
        self.visualize_results(img_path, results, preprocessing_methods)
    
    def test_directory(self, dir_path):
        """Test OCR on all images in a directory"""
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_paths.extend(list(Path(dir_path).glob(f"*{ext}")))
        
        if not image_paths:
            print(f"No images found in {dir_path}")
            return
        
        print(f"Found {len(image_paths)} images to test")
        
        for img_path in image_paths:
            self.test_image(str(img_path))

def main():
    parser = argparse.ArgumentParser(description="Test OCR methods on extracted frames")
    parser.add_argument("--image", type=str, help="Path to a single image to test")
    parser.add_argument("--dir", type=str, help="Path to directory of images to test")
    
    args = parser.parse_args()
    
    tester = OCRTester()
    
    if args.image:
        tester.test_image(args.image)
    elif args.dir:
        tester.test_directory(args.dir)
    else:
        # Default: test all extracted regions
        tester.test_directory("test_data/ocr_regions")

if __name__ == "__main__":
    main()