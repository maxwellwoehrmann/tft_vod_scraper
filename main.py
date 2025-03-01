from src import downloader, gather_vods, mock_db, find_frames, identify_augments
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
                player_frames, bad_frames = find_frames.find_scouting_frames(video_path, "assets/selector_template.png", vod)
                print(player_frames)
                print(bad_frames)       
                
                # os.makedirs('temp/augment_frames', exist_ok=True)
                # for player in player_frames:
                #     player_frames[player].reverse()
                #     frames = player_frames[player]
                #     augment1 = None
                #     augment2 = None
                #     augment3 = None
                #     count = 0
                #     for frame in frames:
                #         print(frame)
                #         image = cv2.imread(frame)
                #         best_x, best_y, pixels = identify_augments.detect_augments(image)
                #         if pixels < 800:
                #             print("Three augments not found")
                #         else:
                #             x, y, w, h = 1300+best_x, 280+best_y, 105, 30
                #             image_roi = image[y:y+h, x:x+w]
                #             cv2.imwrite(f'temp/augment_frames/{player}_{count}.jpg', image_roi)
                #         count += 1

                    

                # results = [
                #     self.matcher.analyze_frame(frame)
                #     for frame in high_res_frames
                # ]
                
                # # 7. Save results
                # self.database.save_results(vod.id, results)
                
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