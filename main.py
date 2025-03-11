from src import downloader, gather_vods, find_frames, identify_placements, detect_and_label_augments, database, cleanup
import logging
import shutil
import os
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tft_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('tft_pipeline')

def process_vod(vod):
    """Process a single VOD and save results to database"""
    try:
        # Store basic game info in database
        database.add_game({
            "game_id": vod["game_id"],
            "match_id": vod["match_id"],
            "vod_url": vod["vod_url"],
            "game_start": vod["game_start"],
            "game_finish": vod["game_finish"],
            "players_list": vod["players"],  # Raw list of players from API
        })
        
        # Download full vod
        logger.info(f"Downloading VOD {vod['game_id']}")
        video_path = downloader.download_vod(vod)

        if not video_path:
            logger.error(f"Failed to download VOD {vod['game_id']}")
            return False
        
        # Extract frames
        logger.info(f"Extracting frames for VOD {vod['game_id']}")
        player_frames, bad_frames, augments, streamer = find_frames.find_scouting_frames(
            video_path, "assets/selector_template.png", vod
        )

        # Detect and label augments
        logger.info(f"Detecting augments for {len(player_frames)} players")
        augment_data = detect_and_label_augments.process_images(player_frames, augments, streamer)

        if not augment_data:
            logger.warning(f"No Augment Data found for VOD {vod['game_id']}")
            cleanup.cleanup_temp_files(video_path, logger)
            return False

        # Check final placements 
        logger.info(f"Getting match placements for {vod['match_id']}")
        placements = identify_placements.get_tft_match_placements(vod["match_id"])
        
        if not placements:
            logger.error(f"Failed to get placements for match {vod['match_id']}")
            cleanup.cleanup_temp_files(video_path, logger)
            return False

        # Save results to database
        logger.info(f"Saving results to database for {vod['game_id']}")
        database.save_results(placements, augment_data, vod["match_id"])
        
        # Cleanup
        #cleanup.cleanup_temp_files(video_path, logger)
        
        logger.info(f"Successfully processed VOD {vod['game_id']}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing VOD {vod['game_id']}: {e}", exc_info=True)
        return False

def run_pipeline(batch_size=100, offset=0):
    """Execute full pipeline"""
    while True:
        try:
            # Connect to database
            database.connect_to_database()
            
            # 1. Get VODs
            logger.info("Fetching recent VODs")
            vods = gather_vods.fetch_recent_vods(batch_size, offset)
            
            # 2. Filter unprocessed using database
            logger.info(f"Retrieved {len(vods)} VODs, filtering known games")
            unprocessed = database.filter_known_games(vods)
            logger.info(f"Found {len(unprocessed)} new VODs to process")

            successful_games = []
            
            # 3. Process each VOD
            for vod in unprocessed:
                success = process_vod(vod)
                if success:
                    successful_games.append(vod)

            logger.info(f"Processed {len(successful_games)} VODs successfully")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        finally:
            # Close database connection
            database.close_connection()
            offset += batch_size

if __name__ == "__main__":
    run_pipeline()