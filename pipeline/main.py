from src import downloader, gather_vods, find_frames, identify_placements, detect_and_label_augments, database, cleanup
from src.utils import logger
import os
import cv2
import argparse

def process_vod(vod, debug_mode=False):
    """Process a single VOD and save results to database"""
    log = logger.get_logger(__name__)
    video_path = None
    
    try:
        # Store basic game info in database
        database.add_game({
            "game_id": vod["game_id"],
            "match_id": vod["match_id"],
            "vod_url": vod["vod_url"],
            "game_start": vod["game_start"],
            "game_finish": vod["game_finish"],
            "players_list": vod["players"],
        })
        
        # Download full vod
        log.info(f"Downloading VOD {vod['game_id']}")
        video_path = downloader.download_vod(vod)

        if not video_path:
            log.error(f"Failed to download VOD {vod['game_id']}")
            return False
        
        # Extract frames with debug mode if enabled
        log.info(f"Extracting frames for VOD {vod['game_id']}")
        player_frames, bad_frames, augments, streamer = find_frames.find_scouting_frames(
            video_path, "assets/selector_template.png", vod, debug_mode=debug_mode
        )

        # Detect and label augments with debug mode if enabled
        log.info(f"Detecting augments for {len(player_frames)} players")
        augment_data = detect_and_label_augments.process_images(
            player_frames, augments, streamer, debug_mode=debug_mode
        )

        if not augment_data:
            log.warning(f"No Augment Data found for VOD {vod['game_id']}")
            cleanup.cleanup_temp_files(video_path)
            return False

        # Check final placements 
        log.info(f"Getting match placements for {vod['match_id']}")
        placements = identify_placements.get_tft_match_placements(vod["match_id"])
        
        if not placements:
            log.error(f"Failed to get placements for match {vod['match_id']}")
            cleanup.cleanup_temp_files(video_path)
            return False

        # Save results to database
        log.info(f"Saving results to database for {vod['game_id']}")
        database.save_results(placements, augment_data, vod["match_id"])
        
        # Cleanup only if not in debug mode
        if not debug_mode:
            cleanup.cleanup_temp_files(video_path)
        else:
            log.info(f"Debug mode enabled - keeping temporary files for {vod['game_id']}")
        
        log.info(f"Successfully processed VOD {vod['game_id']}")
        return True
        
    except Exception as e:
        log.error(f"Error processing VOD {vod['game_id']}: {e}", exc_info=True)
        return False
    finally:
        if video_path and not debug_mode:
            cleanup.cleanup_temp_files(video_path)
            log.info(f"Cleaned up files for VOD {vod['game_id']}")

def run_pipeline(batch_size=100, offset=0, debug_mode=False):
    """Execute full pipeline"""
    log = logger.get_logger(__name__)
    
    if debug_mode:
        log.info("DEBUG MODE ENABLED - detailed diagnostics will be saved")
    
    while True:
        try:
            # Connect to database
            database.connect_to_database()
            
            # 1. Get VODs
            log.info("Fetching recent VODs")
            vods = gather_vods.fetch_recent_vods(batch_size, offset)
            
            # 2. Filter unprocessed using database
            log.info(f"Retrieved {len(vods)} VODs, filtering known games")
            unprocessed = database.filter_known_games(vods)
            log.info(f"Found {len(unprocessed)} new VODs to process")

            successful_games = []
            
            # 3. Process each VOD
            for vod in unprocessed:
                success = process_vod(vod, debug_mode=debug_mode)
                if success:
                    successful_games.append(vod)

            log.info(f"Processed {len(successful_games)} VODs successfully")
            
        except Exception as e:
            log.error(f"Pipeline execution failed: {e}", exc_info=True)
        finally:
            # Close database connection
            database.close_connection()
            offset += batch_size

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TFT VOD Analysis Pipeline")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed diagnostics")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of VODs to process in one batch")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset for VOD retrieval")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Convert log level string to corresponding constant
    log_level_map = {
        "DEBUG": logger.DEBUG,
        "INFO": logger.INFO,
        "WARNING": logger.WARNING,
        "ERROR": logger.ERROR
    }
    log_level = log_level_map.get(args.log_level, logger.INFO)
    
    # Setup logging for the application
    logger.configure_logger(
        log_level=log_level,
        log_dir="logs",
        app_name="tft_pipeline"
    )
    
    log = logger.get_logger(__name__)
    log.info("Starting TFT Pipeline application")
    
    if args.debug:
        log.info("DEBUG MODE ENABLED - Detailed diagnostics will be saved to 'debug' directory")
        # Create debug directory if it doesn't exist
        os.makedirs("debug", exist_ok=True)
    
    # Run the pipeline with command line args
    run_pipeline(
        batch_size=args.batch_size,
        offset=args.offset,
        debug_mode=args.debug
    )