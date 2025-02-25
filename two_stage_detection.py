from src.augment_model_v2.pipeline import AugmentDetectionPipeline
from pathlib import Path
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='TFT Augment Detection: Two-Stage Approach')
    parser.add_argument('--mode', choices=['train_region', 'train_augment', 'detect', 'train_all'], required=True,
                        help='Operation mode: train_region, train_augment, detect, or train_all')
    parser.add_argument('--input', type=str,
                        help='Input directory or image path for detection')
    parser.add_argument('--region_model', type=str, default=None,
                        help='Path to trained region detector model')
    parser.add_argument('--augment_model', type=str, default=None,
                        help='Path to trained augment classifier model')
    parser.add_argument('--augments_dir', type=str, default='assets/augments',
                        help='Directory containing augment images')
    parser.add_argument('--boards_dir', type=str, default='assets/boards',
                        help='Directory containing board images')
    parser.add_argument('--region_crops_dir', type=str, default=None,
                        help='Directory containing real region crops')
    parser.add_argument('--classes_path', type=str, default='assets/augment_classes.txt',
                        help='Path to text file containing augment class names')
    parser.add_argument('--output_dir', type=str, default='model_output/two_stage',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to run models on: cuda, cpu, or mps')
    parser.add_argument('--roi_x', type=int, default=1300,
                        help='X coordinate of ROI')
    parser.add_argument('--roi_y', type=int, default=280,
                        help='Y coordinate of ROI')
    parser.add_argument('--roi_width', type=int, default=130,
                        help='Width of ROI')
    parser.add_argument('--roi_height', type=int, default=100,
                        help='Height of ROI')
    parser.add_argument('--board_crop_x1', type=int, default=380,
                        help='X1 coordinate for board cropping')
    parser.add_argument('--board_crop_y1', type=int, default=130,
                        help='Y1 coordinate for board cropping')
    parser.add_argument('--board_crop_x2', type=int, default=440,
                        help='X2 coordinate for board cropping')
    parser.add_argument('--board_crop_y2', type=int, default=170,
                        help='Y2 coordinate for board cropping')
    parser.add_argument('--force_regenerate', action='store_true',
                        help='Force regeneration of training data')
    
    args = parser.parse_args()
    
    # Define ROI and board crop coordinates
    roi = (args.roi_x, args.roi_y, args.roi_width, args.roi_height)
    board_crop = (args.board_crop_x1, args.board_crop_y1, args.board_crop_x2, args.board_crop_y2)
    
    # Initialize pipeline
    pipeline = AugmentDetectionPipeline(
        region_model_path=args.region_model,
        augment_model_path=args.augment_model,
        output_dir=args.output_dir,
        device=args.device,
        roi=roi,
        augment_templates_dir=args.augments_dir  # Use high-quality augments for verification
    )
    
    if args.mode == 'train_region' or args.mode == 'train_all':
        print("Starting training for region detector...")
        
        # Configure synthetic data generation
        synthetic_config = {
            'augments_dir': args.augments_dir,
            'boards_dir': args.boards_dir,
            'output_dir': os.path.join(args.output_dir, 'region_data'),
            'num_samples': args.num_samples,
            'augment_size': (30, 30),
            'strip_spacing': 1,
            'roi_size': (args.roi_width, args.roi_height),
            'board_crop_coords': board_crop
        }
        
        # Train region detector
        region_model_path = pipeline.train_region_detector(
            synthetic_config=synthetic_config,
            epochs=args.epochs,
            force_regenerate=args.force_regenerate
        )
        
        print(f"Region detector training complete. Model saved to {region_model_path}")
        
        # Update region model path if we're continuing to augment training
        if args.mode == 'train_all':
            pipeline.region_detector = args.region_model = region_model_path
    
    if args.mode == 'train_augment' or args.mode == 'train_all':
        print("Starting training for augment classifier...")
        
        augment_model_path = pipeline.train_augment_classifier(
            augments_dir=args.augments_dir,
            classes_path=args.classes_path,
            num_samples=args.num_samples,
            epochs=args.epochs,
            region_crops_dir=args.region_crops_dir,
            force_regenerate=args.force_regenerate
        )
        
        print(f"Augment classifier training complete. Model saved to {augment_model_path}")
    
    elif args.mode == 'detect':
        if not args.input:
            print("Error: --input is required for detection mode")
            return
        
        if not args.region_model:
            print("Error: --region_model is required for detection mode")
            return
        
        if not args.augment_model:
            print("Warning: --augment_model not provided, will only detect regions without classifying augments")
        
        input_path = Path(args.input)
        if input_path.is_file():
            # Process single image
            print(f"Processing image: {input_path}")
            results = pipeline.process_image(str(input_path))
            
            # Print results
            for i, region in enumerate(results):
                confidence = region['confidence']
                box = region.get('box_absolute', region['box'])
                print(f"Region {i+1}: confidence={confidence:.2f}, box={box}")
                
                if 'augments' in region and region['augments']:
                    print(f"  Augments detected: {len(region['augments'])}")
                    for j, augment in enumerate(region['augments']):
                        print(f"    {j+1}. {augment['class']} (confidence: {augment['confidence']:.2f})")
        
        elif input_path.is_dir():
            # Process directory
            print(f"Processing directory: {input_path}")
            results = pipeline.process_directory(str(input_path))
            
            print(f"Processed {len(results)} images. Results saved to {args.output_dir}")
            
        else:
            print(f"Error: Input path {input_path} does not exist")

if __name__ == "__main__":
    main()