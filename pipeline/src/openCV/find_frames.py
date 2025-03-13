import cv2
import os
import easyocr
from ..utils import string_match, logger
from ..utils.debug import DebugManager
from . import identify_augments

def match_template(roi, template):
    """Returns True if the template is found with sufficient confidence."""
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val >= 0.62, max_loc[0], max_loc[1]  # Return match status and y-coordinate

def find_scouting_frames(video_path, template_path, vod, frame_skip=10, output_dir: str = 'temp/frames', debug_mode=False):
    """
    Find frames that show player scouting to extract augment data
    
    Args:
        video_path: Path to the downloaded VOD
        template_path: Path to the template image for detection
        vod: VOD information dictionary
        frame_skip: Number of frames to skip between checks
        output_dir: Directory to save extracted frames
        debug_mode: Whether to enable debug mode
        
    Returns:
        tuple of: (player_frames, bad_frames, augments, streamer)
    """
    log = logger.get_logger(__name__)
    
    # Initialize debug manager if debug mode is enabled
    debug = DebugManager(debug_enabled=debug_mode)
    if debug_mode:
        debug.set_current_vod(vod['game_id'])
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('temp/augments', exist_ok=True)
    
    log.info(f"Starting frame extraction for game {vod['game_id']}")
    log.debug(f"Video path: {video_path}")
    log.debug(f"Template path: {template_path}")
    log.debug(f"Frame skip interval: {frame_skip}")
    log.debug(f"Debug mode: {debug_mode}")

    reader = easyocr.Reader(['en', 'ch_sim'])
    ja_reader = easyocr.Reader(['ja'])
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

                # Create a debug folder for this frame regardless of whether we find a player
                frame_debug_dir = None
                if debug_mode:
                    # Use a temporary folder name until we know if we found a player
                    frame_debug_dir = debug.create_frame_folder("frame", index, timestamp)
                    # Save full scouting frame
                    debug.save_scouting_frame(frame, frame_debug_dir)
                
                # Process the frame directly instead of loading it again, passing the debug dir
                found, name, used_extended = find_name(frame, match_y, players, reader, ja_reader, debug_mode, debug, frame_debug_dir)
                
                if found:
                    # If we found a player and debug is enabled, rename the folder to the player's name
                    if debug_mode:
                        # Extract ROI for box detection and augment detection
                        augment_roi_x = 1270
                        augment_roi_y = 220
                        augment_roi_width = 160
                        augment_roi_height = 160
                        augment_roi = frame[augment_roi_y:augment_roi_y+augment_roi_height, 
                                        augment_roi_x:augment_roi_x+augment_roi_width]
                        
                        # Save box detection debug
                        debug.save_box_detection(augment_roi, frame_debug_dir)
                        
                        # Rename the folder to include the player name
                        new_dir = os.path.join(os.path.dirname(frame_debug_dir), f"{name}_frame_{index}_t{timestamp:.1f}")
                        try:
                            os.rename(frame_debug_dir, new_dir)
                            frame_debug_dir = new_dir
                        except Exception as e:
                            log.error(f"Failed to rename debug directory: {e}")
                    
                    log.info(f"Identified player: {name}")
                    player_frames[name].append(output_image_path)
                else:
                    # Keep the frame debug dir as "unidentified"
                    if debug_mode:
                        # Rename the folder to indicate unidentified frame
                        new_dir = os.path.join(os.path.dirname(frame_debug_dir), f"unidentified_frame_{index}_t{timestamp:.1f}")
                        try:
                            os.rename(frame_debug_dir, new_dir)
                            frame_debug_dir = new_dir
                        except Exception as e:
                            log.error(f"Failed to rename debug directory: {e}")
                    
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

def find_name(frame, y, players, reader, ja_reader=None, debug_mode=False, debug=None, frame_debug_dir=None):
    """Find and identify player name in the frame"""
    log = logger.get_logger(__name__)
    
    x = 1665  # spare margin for long names
    name_subsection = frame[y:y+48, x:x+148]

    # Create debug visuals for standard OCR region
    if debug_mode and debug is not None and frame_debug_dir is not None:
        debug.save_ocr_debug(name_subsection, players, frame_debug_dir, name_y=y)
        
        # Add detailed debugging information to a text file
        with open(os.path.join(frame_debug_dir, "debug_info.txt"), "w") as f:
            f.write("DEBUGGING PLAYER IDENTIFICATION\n")
            f.write("==============================\n\n")
            
            # 1. Write the exact player list being passed in
            f.write("1. Player List Being Used:\n")
            f.write("-----------------------\n")
            for i, player in enumerate(players):
                f.write(f"  {i+1}. '{player}'\n")
            f.write("\n")

    success, player = read_text(name_subsection, players, reader, ja_reader, debug_mode, frame_debug_dir)
    log.debug(f"First OCR attempt: {'Success' if success else 'Failed'}")
    
    used_extended = False
    if not success:  # it is possible player was leveling up, or starred up a unit while being scouted
        larger_subsection = frame[y-25:y+75, x-70:x+150]
        log.debug("Trying with expanded view")
        
        # Create debug visuals for extended OCR region
        if debug_mode and debug is not None and frame_debug_dir is not None:
            debug.save_ocr_debug(larger_subsection, players, frame_debug_dir, name_y=y, extended=True)
        
        success, player = read_text(larger_subsection, players, reader, ja_reader, debug_mode, frame_debug_dir)
        used_extended = True
        log.debug(f"Second OCR attempt: {'Success' if success else 'Failed'}")

    # Add final result information to debug file
    if debug_mode and frame_debug_dir:
        with open(os.path.join(frame_debug_dir, "debug_info.txt"), "a") as f:
            f.write("\n3. Final Result from find_name():\n")
            f.write("-----------------------------\n")
            f.write(f"  Success: {success}\n")
            f.write(f"  Player: {player if player else 'None'}\n")
            f.write(f"  Used Extended Region: {used_extended}\n")

    return success, player, used_extended

def preprocess_for_ocr(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert back to BGR for OCR compatibility
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return processed

def process_ocr_results(results, players, log):
    """
    Process OCR results and try to match to a player name.
    
    Args:
        results: List of OCR detection results
        players: List of player names to match against
        log: Logger instance
        
    Returns:
        Tuple of (success, player_name)
    """
    for detection in results:
        text = detection[1]
        confidence = detection[2]
        log.debug(f"OCR text: '{text}', confidence: {confidence:.4f}")
        
        success, player = string_match.match_ocr_name(players, text)
        if success:
            log.debug(f"Matched OCR text '{text}' to player '{player}'")
            return True, player
        else:
            log.debug(f"No match found for OCR text: '{text}'")
    
    return False, None

def read_text(image, players, reader, ja_reader=None, debug_mode=False, frame_debug_dir=None):
    """
    Read and match text from image to a known player.
    
    Args:
        image: The image containing potential player name
        players: List of player names to match against
        reader: Primary OCR reader
        ja_reader: Japanese OCR reader (optional)
        debug_mode: Whether debug mode is enabled
        frame_debug_dir: Directory to save debug information
        
    Returns:
        Tuple of (success, player_name)
    """
    log = logger.get_logger(__name__)
    
    # Pre-process the image
    processed_image = preprocess_for_ocr(image)
    
    # Try with primary reader
    results = reader.readtext(processed_image)
    log.debug(f"Primary OCR found {len(results)} text regions")
    
    # Debug OCR results
    if debug_mode and frame_debug_dir:
        with open(os.path.join(frame_debug_dir, "debug_info.txt"), "a") as f:
            f.write("\n2. OCR Results and String Matching:\n")
            f.write("-------------------------------\n")
            f.write("\nPrimary OCR Results:\n")
            for i, detection in enumerate(results):
                text = detection[1]
                confidence = detection[2]
                f.write(f"  Result {i+1}: '{text}' (confidence: {confidence:.4f})\n")
                
                # Call string matching and record results
                success, matched_player = string_match.match_ocr_name(players, text)
                f.write(f"    String Match Result: success={success}, player='{matched_player}'\n")
    
    # Process primary OCR results
    success, name = process_ocr_results(results, players, log)
    
    # If no match found, try Japanese reader if available
    if not success and ja_reader is not None:
        log.debug("Attempting text recognition with Japanese OCR reader")
        ja_results = ja_reader.readtext(processed_image)
        log.debug(f"Japanese OCR found {len(ja_results)} text regions")
        
        # Debug Japanese OCR results
        if debug_mode and frame_debug_dir:
            with open(os.path.join(frame_debug_dir, "debug_info.txt"), "a") as f:
                f.write("\nJapanese OCR Results:\n")
                for i, detection in enumerate(ja_results):
                    text = detection[1]
                    confidence = detection[2]
                    f.write(f"  Result {i+1}: '{text}' (confidence: {confidence:.4f})\n")
                    
                    # Call string matching and record results
                    success, matched_player = string_match.match_ocr_name(players, text)
                    f.write(f"    String Match Result: success={success}, player='{matched_player}'\n")
        
        # Process Japanese OCR results
        success, name = process_ocr_results(ja_results, players, log)
    
    if not success:
        log.debug("No matching player found in any OCR results")
    
    return success, name