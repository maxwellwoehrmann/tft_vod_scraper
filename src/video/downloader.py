import yt_dlp
import sys
import argparse
import subprocess
import os
import ffmpeg
import time
from datetime import datetime, timedelta
from ..utils import logger

def download_vod(vod_info: dict, quality='1920p'):
    """
    Download specific portion of Twitch VOD using direct yt-dlp command
    
    Args:
        vod_info: Dictionary containing VOD information
        quality: Desired video quality
        
    Returns:
        Path to downloaded video file or None if download failed
    """
    log = logger.get_logger(__name__)
    
    # Extract base VOD URL (remove timestamp)
    base_url = vod_info['vod_url'].split('?')[0]
    
    # Create temp directory if it doesn't exist
    os.makedirs('temp', exist_ok=True)
    
    output_path = f"temp/game-{vod_info['game_id']}.mp4"
    
    cmd = [
        "yt-dlp",
        "-S", "vcodec:h264,fps,res,acodec:m4a",
        "--download-sections", f"*{vod_info['game_start']}-{vod_info['game_finish']}",
        "-o", output_path,
        "-N", "16",
        base_url
    ]
    
    log.info(f"Downloading VOD {vod_info['game_id']} segment from {vod_info['game_start']} to {vod_info['game_finish']}")
    log.debug(f"Download command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Check if file was actually downloaded
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            log.info(f"Successfully downloaded VOD {vod_info['game_id']} ({file_size:.2f} MB)")
            return output_path
        else:
            log.error(f"Download process completed but file not found: {output_path}")
            return None
    except subprocess.CalledProcessError as e:
        log.error(f"Error during download: {e}")
        log.debug(f"Command stderr: {e.stderr}")
        return None
    except Exception as e:
        log.error(f"Unexpected error during download: {e}", exc_info=True)
        return None