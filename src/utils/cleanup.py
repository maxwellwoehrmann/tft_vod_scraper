import os, shutil

def cleanup_temp_files(video_path, logger):
    """Clean up temporary files after processing"""
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"Removed video file: {video_path}")
        
        # Clean up temp directory if it exists
        if os.path.exists('temp'):
            for subdir in ['frames', 'augments']:
                path = os.path.join('temp', subdir)
                if os.path.exists(path):
                    shutil.rmtree(path)
                    logger.info(f"Removed directory: {path}")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")