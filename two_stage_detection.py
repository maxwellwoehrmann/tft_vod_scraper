from src.augment_model_v2.pipeline import AugmentDetectionPipeline
from pathlib import Path
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='TFT Augment Detection: Two-Stage Approach')
    parser.add_argument('--mode', choices=['train', 'detect'], required=True,
                        help='Operation mode: train or detect')
    parser.add_argument('--input', type=str,
                        help='Input directory or image path for detection')
    parser.add_argument('--region_model', type=str, default=None,
                        help='Path to trained region detector model (for detection mode)')
    parser.add_argument('--augments_dir', type=str, default='assets/augments',
                        help='Directory containing augment images (for training mode)')
    parser.add_argument('--boards_dir', type=str, default='assets/boards',
                        help='Directory containing board images (for training mode)')
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
    
    args = parser.parse_args()
    
    # Define ROI
    roi = (args.roi_x, args.roi_y, args.roi_width, args.roi_height)
    board_crop = (args.board_crop_x1, args.board_crop_y1, args.board_crop_x2, args.board_crop_y2)
    
    # Initialize pipeline
    pipeline = AugmentDetectionPipeline(
        region_model_path=args.region_model,
        output_dir=args.output_dir,
        device=args.device,
        roi=roi
    )
    
    if args.mode == 'train':
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
        model_path = pipeline.train_region_detector(
            synthetic_config=synthetic_config,
            epochs=args.epochs
        )
        
        print(f"Region detector training complete. Model saved to {model_path}")
        
    elif args.mode == 'detect':
        if not args.input:
            print("Error: --input is required for detection mode")
            return
        
        input_path = Path(args.input)
        if input_path.is_file():
            # Process single image
            print(f"Processing image: {input_path}")
            regions = pipeline.process_image(str(input_path))
            
            for i, region in enumerate(regions):
                confidence = region['confidence']
                box = region.get('box_absolute', region['box'])
                print(f"Region {i+1}: confidence={confidence:.2f}, box={box}")
                
        elif input_path.is_dir():
            # Process directory
            print(f"Processing directory: {input_path}")
            results = pipeline.process_directory(str(input_path))
            
            print(f"Processed {len(results)} images. Results saved to {args.output_dir}")
            
        else:
            print(f"Error: Input path {input_path} does not exist")

if __name__ == "__main__":
    main()