from src import downloader, gather_vods, mock_db, find_frames, identify_placements, detect_and_label_augments, database
import logging
import shutil
import os
import cv2

class Pipeline:
    #def __init__(self, config_path):
        #self.twitch_api = TwitchAPI(self.config['api_key'])
        #self.database = GameDatabase(self.config['db_path'])
        #self.downloader = VideoDownloader(self.config['output_dir'])
        #self.frame_extractor = FrameExtractor(self.config['template_path'])
        #self.matcher = TemplateMatcher(self.config['templates_dir'])
    
    def run(self):
        """Execute full pipeline"""
        # 1. Get VODs
        vods = gather_vods.fetch_recent_vods(10)
        
        # 2. Filter unprocessed
        unprocessed = mock_db.filter_known_games(vods)

        successful_games = []
        
        for vod in unprocessed:
            try:
                # 3. Download full vod
                video_path = downloader.download_vod(vod)

                if not video_path:
                    print(f"Failed to download VOD {vod['game_id']}")
                    continue
                
                #4. Extract frames
                player_frames, bad_frames, augments = find_frames.find_scouting_frames(video_path, "assets/selector_template.png", vod)

                #5. Detect and label augments
                print(player_frames)
                print(augments)
                augment_data = detect_and_label_augments.process_images(player_frames, augments)

                if not augment_data:
                    print(f"No Augment Data found for VOD {vod['game_id']}")
                    continue

                #6. Check final placements 
                placements = identify_placements.get_tft_match_placements(vod["match_id"])
                print(vod['players'])
                print(placements)

                # 7. Save results
                mock_db.save_results(placements, augment_data, "data/augment_performance.json")
                
                # # 8. Cleanup
                # self._cleanup(video_path, frames)
                
                # # 9. Mark processed
                # self.database.mark_game_processed(vod.id)

                successful_games.append(vod)
                
            except Exception as e:
                logging.error(f"Error processing VOD {vod['game_id']}: {e}")
                continue

            #shutil.rmtree('temp') #clean-up all temp files after processing game
    
        if successful_games:
            mock_db.add_games_to_csv(successful_games)

if __name__ == "__main__":
    # Setup

    #setup_logging()
    
    # Initialize and run pipeline
    #pipeline = Pipeline(config)

    pipeline = Pipeline()
    pipeline.run()