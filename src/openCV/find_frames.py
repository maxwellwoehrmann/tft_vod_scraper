import cv2
import os
import easyocr
from ..utils import string_match

def match_template(roi, template):
    """Returns True if the template is found with sufficient confidence."""
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val >= 0.62, max_loc[0], max_loc[1]  # Return match status and y-coordinate

def find_scouting_frames(video_path, template_path, vod, frame_skip=10, output_dir: str = 'temp/frames'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('temp/augments', exist_ok=True)

    reader = easyocr.Reader(['en', 'ja'])

    game_id = vod['game_id']
    players = vod['players']

    player_frames = dict()
    for player in players:
        player_frames[player] = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    template = cv2.imread(template_path)  # Read in color
    reroll_template = cv2.imread("templates/reroll_template.jpg")

    frame_number = 0
    last_match_y = None  # Store the y-coordinate of the last match
    index = 0

    augment_index = 1
    augment_cooldown = 0

    bad_frames = []

    # Reroll ROI Coordinates
    x, y, w, h = 900, 830, 120, 65

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % (fps*60) == 0:
            print(f'Minute: {frame_number / (fps*60)}')

        if frame_number % frame_skip != 0:  # Skip unnecessary frames
            frame_number += 1
            augment_cooldown -= 1
            continue

        

        #check for reroll frame first (augment selection screen)
        if augment_cooldown <= 0:
            x, y, w, h = 900, 830, 120, 65
            reroll_roi = frame[y:y+h, x:x+w]
            reroll_match, _, _ = match_template(reroll_roi, reroll_template)
            if reroll_match:
                x, y, w, h = 895, 330, 140, 140
                augment_roi = frame[y:y+h, x:x+w]

                output_image_path = f"temp/augments/augment_{augment_index}.jpg"
                cv2.imwrite(output_image_path, augment_roi)

                print(f"Found Augment {augment_index}")

                augment_index += 1
                augment_cooldown = fps * 60 * 5 #sleep this function for 5 minutes

        roi = frame[60:850, -22:]  # Extract rightmost 22 pixels (all rows in color frame)

        # Check both templates
        match, _, y, = match_template(roi, template)
        y += 60 #make up for the offset incurred by the ROI

        # Process potential matches
        save_match = False
        match_y = None

        if match:
            if last_match_y is None or abs(y - last_match_y) > 3:
                save_match = True
                match_y = y
        
        if save_match:
            timestamp = round(frame_number / fps, 3)
            last_match_y = match_y

            print(f"Match found at {timestamp:.3f} seconds (y={match_y})")

            output_image_path = f"{output_dir}/frame_{game_id}_{index}.jpg"
            cv2.imwrite(output_image_path, frame)

            # Process the frame directly instead of loading it again
            found, name = find_name(frame, match_y, players, reader)
            if found:
                player_frames[name].append(output_image_path)
            else:
                bad_frames.append(output_image_path)
            index += 1

        frame_number += 1
        augment_cooldown -= 1

    cap.release()

    return player_frames, bad_frames

def find_name(frame, y, players, reader):
    x = 1665 #spare margin for long names

    name_subsection = frame[y:y+48, x:x+148]

    success, player = read_text(name_subsection, players, reader)

    if not success: #it is possible player was leveling up, or starred up a unit while being scouted.
        larger_subsection = frame[y-25:y+75, x-70:x+150]
        cv2.imwrite("temp/expanded_view.jpg", larger_subsection) #debug larger section
        #try again with a larger field of view
        success, player = read_text(larger_subsection, players, reader)

    return success, player

def read_text(image, players, reader):
    # Read the image
    results = reader.readtext(image)
    
    success = False
    name = None
    # Returns list of (bounding box, text, confidence)
    for detection in results: #check all strings found in image, may contain multiple texts for edge cases: starring up unit, leveling up
        print(f"Text: {detection[1]}, Confidence: {detection[2]}")
        success, player = string_match.match_ocr_name(players, detection[1])
        if success:
            name = player
            break

    return success, name
