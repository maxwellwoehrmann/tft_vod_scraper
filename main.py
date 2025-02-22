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
        vods = gather_vods.fetch_recent_vods()
        
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
                player_frames, bad_frames = find_frames.find_scouting_frames(video_path, "templates/selector_template.png", vod)
                print(player_frames)
                print(bad_frames)     
                #player_frames = {'riri': ['temp/frames/frame_496616530_23.jpg'], 'ティエルノ': ['temp/frames/frame_496616530_24.jpg', 'temp/frames/frame_496616530_28.jpg', 'temp/frames/frame_496616530_29.jpg', 'temp/frames/frame_496616530_30.jpg', 'temp/frames/frame_496616530_31.jpg', 'temp/frames/frame_496616530_38.jpg', 'temp/frames/frame_496616530_55.jpg', 'temp/frames/frame_496616530_58.jpg', 'temp/frames/frame_496616530_75.jpg', 'temp/frames/frame_496616530_78.jpg', 'temp/frames/frame_496616530_80.jpg'], 'absjkfbfhjjk': ['temp/frames/frame_496616530_10.jpg', 'temp/frames/frame_496616530_16.jpg', 'temp/frames/frame_496616530_20.jpg', 'temp/frames/frame_496616530_21.jpg', 'temp/frames/frame_496616530_44.jpg', 'temp/frames/frame_496616530_56.jpg', 'temp/frames/frame_496616530_57.jpg', 'temp/frames/frame_496616530_60.jpg', 'temp/frames/frame_496616530_62.jpg', 'temp/frames/frame_496616530_64.jpg', 'temp/frames/frame_496616530_66.jpg', 'temp/frames/frame_496616530_69.jpg'], 'ajuna': ['temp/frames/frame_496616530_8.jpg', 'temp/frames/frame_496616530_19.jpg', 'temp/frames/frame_496616530_35.jpg', 'temp/frames/frame_496616530_42.jpg', 'temp/frames/frame_496616530_50.jpg', 'temp/frames/frame_496616530_54.jpg', 'temp/frames/frame_496616530_59.jpg'], 'tteru': ['temp/frames/frame_496616530_7.jpg', 'temp/frames/frame_496616530_13.jpg', 'temp/frames/frame_496616530_22.jpg', 'temp/frames/frame_496616530_26.jpg', 'temp/frames/frame_496616530_34.jpg', 'temp/frames/frame_496616530_65.jpg', 'temp/frames/frame_496616530_71.jpg'], 'gummmmmmmmi': ['temp/frames/frame_496616530_6.jpg', 'temp/frames/frame_496616530_14.jpg', 'temp/frames/frame_496616530_17.jpg', 'temp/frames/frame_496616530_48.jpg', 'temp/frames/frame_496616530_53.jpg', 'temp/frames/frame_496616530_61.jpg', 'temp/frames/frame_496616530_70.jpg', 'temp/frames/frame_496616530_72.jpg'], 'toyschory': ['temp/frames/frame_496616530_5.jpg', 'temp/frames/frame_496616530_12.jpg', 'temp/frames/frame_496616530_25.jpg'], 'hizashi': ['temp/frames/frame_496616530_2.jpg', 'temp/frames/frame_496616530_4.jpg', 'temp/frames/frame_496616530_11.jpg', 'temp/frames/frame_496616530_18.jpg', 'temp/frames/frame_496616530_37.jpg', 'temp/frames/frame_496616530_40.jpg', 'temp/frames/frame_496616530_45.jpg', 'temp/frames/frame_496616530_47.jpg', 'temp/frames/frame_496616530_68.jpg']}
                    
                
                os.makedirs('temp/augment_frames', exist_ok=True)
                for player in player_frames:
                    player_frames[player].reverse()
                    frames = player_frames[player]
                    augment1 = None
                    augment2 = None
                    augment3 = None
                    count = 0
                    for frame in frames:
                        print(frame)
                        image = cv2.imread(frame)
                        best_x, best_y, pixels = identify_augments.detect_augments(image)
                        if pixels < 800:
                            print("Three augments not found")
                        else:
                            x, y, w, h = 1300+best_x, 280+best_y, 100, 30
                            image_roi = image[y:y+h, x:x+w]
                            cv2.imwrite(f'temp/augment_frames/{player}_{count}.jpg', image_roi)
                        count += 1



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