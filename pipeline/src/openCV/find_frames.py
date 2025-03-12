import cv2
import os
import easyocr
from ..utils import string_match, logger
from . import identify_augments

def match_template(roi, template):
    """Returns True if the template is found with sufficient confidence."""
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val >= 0.62, max_loc[0], max_loc[1]  # Return match status and y-coordinate

def find_scouting_frames(video_path, template_path, vod, frame_skip=10, output_dir: str = 'temp/frames'):
    """
    Find frames that show player scouting to extract augment data
    
    Args:
        video_path: Path to the downloaded VOD
        template_path: Path to the template image for detection
        vod: VOD information dictionary
        frame_skip: Number of frames to skip between checks
        output_dir: Directory to save extracted frames
        
    Returns:
        tuple of: (player_frames, bad_frames, augments, streamer)
    """
    log = logger.get_logger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('temp/augments', exist_ok=True)
    
    log.info(f"Starting frame extraction for game {vod['game_id']}")
    log.debug(f"Video path: {video_path}")
    log.debug(f"Template path: {template_path}")
    log.debug(f"Frame skip interval: {frame_skip}")

    reader = easyocr.Reader(['en', 'ch_sim'])
    log.debug("Initialized OCR reader")

    game_id = vod['game_id']
    players = vod['players']
    log.info(f"Processing {len(players)} players")

    player_frames = dict()
    for player in players:
        player_frames[player] = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error(f"Failed to open video file: {video_path}")
            return {}, [], [], None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        
        log.info(f"Video loaded: {fps:.2f} FPS, {total_frames} frames, {duration_sec:.2f} seconds")
        
        template = cv2.imread(template_path)  # Read in color
        if template is None:
            log.error(f"Failed to load template image: {template_path}")
            cap.release()
            return {}, [], [], None
        
        log.debug(f"Loaded template with shape {template.shape}")
    except Exception as e:
        log.error(f"Error initializing video processing: {e}", exc_info=True)
        return {}, [], [], None

    frame_number = 0
    last_match_y = None  # Store the y-coordinate of the last match
    index = 0

    bad_frames = []
    augments = []

    streamer_frames = []
    streamer_cooldown = int(fps*60*17) #only start capturing streamer frames after 17 minutes, avg time for 3rd augment arrival
    log.debug(f"Streamer frame capture will begin after {streamer_cooldown/fps:.2f} seconds")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                log.debug("End of video reached")
                break

            if frame_number % (fps*60) == 0:
                log.info(f'Processing minute: {frame_number / (fps*60):.1f} ({frame_number}/{total_frames} frames)')

            if frame_number % frame_skip != 0:  # Skip unnecessary frames
                frame_number += 1
                streamer_cooldown -= 1
                continue

            #roi for scouting indicator
            roi = frame[60:850, -22:]  # Extract rightmost 22 pixels (all rows in color frame)

            # Check template
            match, _, y = match_template(roi, template)
            y += 60 #make up for the offset incurred by the ROI

            # Process potential matches
            save_match = False
            match_y = None

            if match:
                if last_match_y is None or abs(y - last_match_y) > 3:
                    save_match = True
                    match_y = y
                    log.debug(f"Template match found at y={y}")
            else:
                # Take screenshots of streamer's screen at regular intervals after cooldown period
                if streamer_cooldown <= 0:
                    output_image_path = f"{output_dir}/frame_{game_id}_{index}.jpg"
                    cv2.imwrite(output_image_path, frame)
                    streamer_frames.append(output_image_path)
                    log.debug(f"Saved streamer frame: {output_image_path}")
                    streamer_cooldown = int(fps * 15) #take a screenshot of the streamers screen every 15 seconds
                    index += 1
            
            if save_match:
                timestamp = round(frame_number / fps, 3)
                last_match_y = match_y

                log.info(f"Match found at {timestamp:.3f} seconds (y={match_y})")

                output_image_path = f"{output_dir}/frame_{game_id}_{index}.jpg"
                cv2.imwrite(output_image_path, frame)
                log.debug(f"Saved match frame: {output_image_path}")

                # Process the frame directly instead of loading it again
                found, name = find_name(frame, match_y, players, reader)
                if found:
                    log.info(f"Identified player: {name}")
                    player_frames[name].append(output_image_path)
                else:
                    log.warning(f"Failed to identify player in frame {index}")
                    bad_frames.append(output_image_path)
                index += 1

            frame_number += 1
            streamer_cooldown -= 1

    except Exception as e:
        log.error(f"Error during frame processing: {e}", exc_info=True)
    finally:
        cap.release()
        log.info(f"Processed {frame_number} frames, found {index} matches")

    streamer = None #find the player with no frames, this should be the streamer
    corrupt = False
    
    # Find the streamer (player with no frames)
    for player in player_frames:
        if len(player_frames[player]) == 0:
            if not streamer:
                streamer = player
                log.info(f"Identified streamer as: {streamer}")
            else: 
                #if we find 2 players without frames, this is a big issue - store no ones data for now
                log.error(f"Multiple players have no frames - potential data corruption: {streamer} and {player}")
                corrupt = True
    
    if corrupt:
        log.warning("Data may be corrupted, multiple players with no frames detected")
    elif streamer:
        player_frames[streamer] = streamer_frames
        log.info(f"Assigned {len(streamer_frames)} frames to streamer {streamer}")

    # Log summary of frames found
    for player, frames in player_frames.items():
        log.info(f"Player {player}: {len(frames)} frames collected")
    log.info(f"Bad frames: {len(bad_frames)}")

    return player_frames, bad_frames, augments, streamer

def find_name(frame, y, players, reader):
    """Find and identify player name in the frame"""
    log = logger.get_logger(__name__)
    
    x = 1665 #spare margin for long names

    name_subsection = frame[y:y+48, x:x+148]

    success, player = read_text(name_subsection, players, reader)
    log.debug(f"First OCR attempt: {'Success' if success else 'Failed'}")

    if not success: #it is possible player was leveling up, or starred up a unit while being scouted.
        larger_subsection = frame[y-25:y+75, x-70:x+150]
        # For debugging: save the expanded view
        #cv2.imwrite("temp/expanded_view.jpg", larger_subsection)
        log.debug("Trying with expanded view")
        #try again with a larger field of view
        success, player = read_text(larger_subsection, players, reader)
        log.debug(f"Second OCR attempt: {'Success' if success else 'Failed'}")

    return success, player

def read_text(image, players, reader):
    """Read and match text from image to a known player"""
    log = logger.get_logger(__name__)
    
    # Read the image
    results = reader.readtext(image)
    
    success = False
    name = None
    
    log.debug(f"OCR found {len(results)} text regions")
    
    # Returns list of (bounding box, text, confidence)
    for detection in results: #check all strings found in image, may contain multiple texts for edge cases: starring up unit, leveling up
        text = detection[1]
        confidence = detection[2]
        log.debug(f"OCR text: '{text}', confidence: {confidence:.4f}")
        
        success, player = string_match.match_ocr_name(players, text)
        if success:
            log.debug(f"Matched OCR text '{text}' to player '{player}'")
            name = player
            break
        else:
            log.debug(f"No match found for OCR text: '{text}'")

    if not success:
        log.debug("No matching player found in OCR results")
        
    return success, name