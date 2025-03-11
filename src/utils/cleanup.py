import os
import shutil
from ..utils import logger

def cleanup_temp_files(video_path, log=None):
    """
    Clean up temporary files after processing
    
    Args:
        video_path: Path to the downloaded video file
        log: Optional logger instance (if not provided, will create one)
    """
    if log is None:
        log = logger.get_logger(__name__)
        
    try:
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
            os.remove(video_path)
            log.info(f"Removed video file: {video_path} ({file_size:.2f} MB)")
        else:
            log.warning(f"Video file not found during cleanup: {video_path}")
        
        # Clean up temp directory if it exists
        temp_dirs = ['temp/frames', 'temp/augments']
        for path in temp_dirs:
            if os.path.exists(path):
                num_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
                shutil.rmtree(path)
                log.info(f"Removed directory: {path} containing {num_files} files")
    except Exception as e:
        log.warning(f"Error during cleanup: {e}", exc_info=True)