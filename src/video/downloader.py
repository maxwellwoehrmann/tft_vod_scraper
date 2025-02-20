import yt_dlp
import sys
import argparse
import subprocess
import os
import ffmpeg
from datetime import datetime, timedelta

def download_low_res(vod_info: dict, quality='160p'):
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
        "-f", "360p",
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

def download_frames(vod_info: dict, frame_timestamps: list, output_dir: str = 'temp/frames'):
    """
    Download high-resolution frames from a VOD at specific timestamps.
    
    Args:
        vod_info (dict): Dictionary containing VOD information including game_start time
        frame_timestamps (list): List of seconds elapsed from game start
        output_dir (str): Directory to save the frames
    
    Returns:
        list: Paths to the saved frame images
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_frames = []
    base_url = vod_info['vod_url'].split('?')[0]

    game_start = datetime.strptime(vod_info['game_start'], '%H:%M:%S')
    
    for i, timestamp in enumerate(frame_timestamps):
        # Calculate actual VOD timestamp
        frame_time = game_start + timedelta(seconds=timestamp)
        # Add 0.1 seconds to get just enough video for one frame
        end_time = frame_time + timedelta(seconds=0.001)
        
        temp_video_path = f"temp/frame_{vod_info['game_id']}_{i}.mp4"
        output_image_path = f"{output_dir}/frame_{vod_info['game_id']}_{i}.jpg"
        
        # Format timestamps for yt-dlp with millisecond precision
        frame_time_str = frame_time.strftime('%H:%M:%S.%f')[:12]  # Get HH:MM:SS.mmm
        end_time_str = (frame_time + timedelta(milliseconds=1)).strftime('%H:%M:%S.%f')[:12]
        
        # Download short high-quality clip
        cmd = [
            "yt-dlp",
            "-S", "res:1920,vcodec:h264,fps,acodec:m4a",  # Prioritize 1920p
            "--download-sections", f"*{frame_time_str}-{end_time_str}",
            "-o", temp_video_path,
            "-N", "16",
            base_url
        ]
        
        try:
            # Download the video snippet
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Extract the first frame using ffmpeg
            ffmpeg.input(temp_video_path, ss=0).output(output_image_path, vframes=1).run(overwrite_output=True)
            saved_frames.append(output_image_path)
            
            # Cleanup temporary video
            os.remove(temp_video_path)
            
        except subprocess.CalledProcessError as e:
            print(f"Error downloading frame at timestamp {timestamp}:\n{e.stderr}")
            continue
        except Exception as e:
            print(f"Error processing frame at timestamp {timestamp}: {str(e)}")
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            continue
            
    return saved_frames