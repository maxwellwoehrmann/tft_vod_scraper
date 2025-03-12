# extract_test_frames.py
import os
import cv2
import yt_dlp
import subprocess
import argparse
import json
from pathlib import Path

# Create directories
os.makedirs("test_data/frames", exist_ok=True)
os.makedirs("test_data/ocr_regions", exist_ok=True)

def download_vod(vod_url, start_time, end_time, output_path="test_data/test_vod.mp4"):
    """Download a specific portion of a VOD"""
    print(f"Downloading VOD segment from {start_time} to {end_time}")
    
    cmd = [
        "yt-dlp",
        "-S", "vcodec:h264,fps,res,acodec:m4a",
        "--download-sections", f"*{start_time}-{end_time}",
        "-o", output_path,
        vod_url
    ]
    
    try:
        subprocess.run(cmd, check=True)
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Successfully downloaded VOD ({size_mb:.2f} MB)")
            return output_path
        else:
            print(f"Download completed but file not found: {output_path}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e}")
        return None

def extract_frame_timestamps(log_path):
    """Extract timestamps of frames where OCR failed from logs"""
    failed_timestamps = []
    successful_timestamps = {}
    
    timestamp = None
    expecting_result = False
    
    with open(log_path, 'r') as f:
        for line in f:
            if "Match found at" in line:
                # Extract timestamp
                try:
                    time_part = line.split("Match found at ")[1].split(" seconds")[0]
                    timestamp = float(time_part)
                    expecting_result = True  # Now expect a result line
                except (IndexError, ValueError):
                    continue
            elif expecting_result and "Failed to identify player in frame" in line:
                # Extract frame number
                try:
                    frame_part = line.split("Failed to identify player in frame ")[1]
                    frame_num = int(frame_part.strip())  # Get just the number part
                    failed_timestamps.append((timestamp, frame_num))
                except (IndexError, ValueError):
                    continue
                expecting_result = False  # We got our result
            elif expecting_result and "Identified player:" in line:
                # Extract player name
                try:
                    player = line.split("Identified player: ")[1].strip()
                    successful_timestamps[timestamp] = player
                except (IndexError, ValueError):
                    continue
                expecting_result = False  # We got our result
    
    return failed_timestamps, successful_timestamps

def extract_frames(video_path, timestamps, output_dir="test_data/frames"):
    """Extract frames at specific timestamps"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Video info: {fps} FPS, {frame_count} frames, {duration:.2f} seconds")
    
    extracted_frames = []
    
    for timestamp, frame_num in timestamps:
        # Calculate frame number from timestamp
        frame_position = int(timestamp * fps)
        
        # Set position and read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()
        
        if ret:
            # Save full frame
            frame_path = f"{output_dir}/frame_{frame_num}_ts_{timestamp:.1f}.jpg"
            cv2.imwrite(frame_path, frame)
            
            # Extract and save ROI for player name (adjust these coordinates as needed)
            # Let's save multiple ROIs to try different regions
            regions = {
                "name_area": (1600, 180, 1850, 250)  # Potential name area
            }
            
            for region_name, (x1, y1, x2, y2) in regions.items():
                try:
                    roi = frame[y1:y2, x1:x2]
                    roi_path = f"test_data/ocr_regions/frame_{frame_num}_{region_name}.jpg"
                    cv2.imwrite(roi_path, roi)
                except Exception as e:
                    print(f"Error extracting ROI {region_name} from frame {frame_num}: {e}")
            
            extracted_frames.append((frame_num, timestamp, frame_path))
            print(f"Extracted frame {frame_num} at timestamp {timestamp:.2f}s")
        else:
            print(f"Failed to extract frame at timestamp {timestamp:.2f}s")
    
    cap.release()
    
    # Save metadata
    metadata = {
        "video_path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "extracted_frames": extracted_frames
    }
    
    with open("test_data/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return extracted_frames

def main():
    parser = argparse.ArgumentParser(description="Extract frames where OCR failed for testing")
    parser.add_argument("--game_id", type=str, required=True, help="Game ID to analyze")
    parser.add_argument("--log_file", type=str, required=True, help="Path to log file")
    parser.add_argument("--vod_url", type=str, required=True, help="Twitch VOD URL")
    parser.add_argument("--start_time", type=str, required=True, help="Start time (HH:MM:SS)")
    parser.add_argument("--end_time", type=str, required=True, help="End time (HH:MM:SS)")
    
    args = parser.parse_args()
    
    # 1. Download VOD
    video_path = download_vod(args.vod_url, args.start_time, args.end_time)
    if not video_path:
        print("Failed to download VOD. Exiting.")
        return
    
    # 2. Extract timestamps of frames where OCR failed
    failed_timestamps, successful_timestamps = extract_frame_timestamps(args.log_file)
    
    print(f"Found {len(failed_timestamps)} frames with failed OCR")
    print(f"Found {len(successful_timestamps)} frames with successful OCR")
    
    # 3. Extract frames
    print("Extracting frames with failed OCR...")
    extract_frames(video_path, failed_timestamps)
    
    # 4. Also extract a few frames with successful OCR for comparison
    print("Extracting a few frames with successful OCR for comparison...")
    successful_items = list(successful_timestamps.items())[:5]  # Just get 5 examples
    extract_frames(video_path, [(ts, i) for i, (ts, _) in enumerate(successful_items)], 
                  output_dir="test_data/good_frames")
    
    # 5. Save player names for reference
    with open("test_data/players.json", "w") as f:
        player_data = {
            "game_id": args.game_id,
            "successful_ocr": successful_timestamps,
            "players": list(set(successful_timestamps.values()))
        }
        json.dump(player_data, f, indent=2)
    
    print(f"Extracted {len(failed_timestamps)} frames for OCR testing")
    print("Done! Run test_ocr.py to experiment with OCR improvements")

if __name__ == "__main__":
    main()