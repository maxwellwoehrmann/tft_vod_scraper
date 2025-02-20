from src import downloader, gather_vods, mock_db, find_frames
import logging 

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
                # 3. Download low-res
                video_path = downloader.download_low_res(vod)

                if not video_path:
                    print(f"Failed to download VOD {vod['game_id']}")
                    continue
                
                # 4. Extract frames
                frames = find_frames.find_scouting_frames(video_path, "templates/template_360_1.png", "templates/template_360_2.png", frame_skip=10)
            
                if frames:
                    high_res_frames = downloader.download_frames(vod, frames)
                    print(f"Downloaded {len(high_res_frames)} high-resolution frames")
                else:
                    print(f"No frames found for VOD {vod['game_id']}")
                    continue
                
                # 5. Discern players
                users = dict()
                high_res_frames.reverse()
                for frame in high_res_frames:
                    found, name = find_frames.find_name(frame, "templates/template_1980_1.png")
                    if found:
                        if name in users:
                            if len(users[name]) < 3:
                                users[name].append(frame)
                        else:
                            users[name] = [frame]
                
                print(users)
                
                    

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
    
        if successful_games:
            mock_db.add_games_to_csv(successful_games)

if __name__ == "__main__":
    # Setup

    #setup_logging()
    
    # Initialize and run pipeline
    #pipeline = Pipeline(config)

    pipeline = Pipeline()
    pipeline.run()
