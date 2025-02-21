import yt_dlp
import sys
import argparse
import subprocess
import os
import ffmpeg
import time
from datetime import datetime, timedelta

def download_vod(vod_info: dict, quality='1920p'):
    """
    Download specific portion of Twitch VOD using direct yt-dlp command
    """
    # Extract base VOD URL (remove timestamp)
    base_url = vod_info['vod_url'].split('?')[0]
    
    # Create temp directory if it doesn't exist
    os.makedirs('temp', exist_ok=True)
    
    cmd = [
        "yt-dlp",
        "-S", "vcodec:h264,fps,res,acodec:m4a",
        "--download-sections", f"*{vod_info['game_start']}-{vod_info['game_finish']}",
        "-o", f"temp/game-{vod_info['game_id']}.mp4",
        "-N", "16",
        base_url
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return f"temp/game-{vod_info['game_id']}.mp4"
    except subprocess.CalledProcessError as e:
        print(f"Error during download:\n{e.stderr}")
        return None