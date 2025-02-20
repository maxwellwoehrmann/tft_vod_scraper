import cv2
import os
import easyocr

def match_template(roi, template):
    """Returns True if the template is found with sufficient confidence."""
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val >= 0.7, max_loc[0], max_loc[1]  # Return match status and y-coordinate

def find_scouting_frames(video_path, template1_path, template2_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    template1 = cv2.imread(template1_path)  # Read in color
    template2 = cv2.imread(template2_path)  # Read in color

    timestamps = []
    frame_number = 0
    last_match_y = None  # Store the y-coordinate of the last match

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip != 0:  # Skip unnecessary frames
            frame_number += 1
            continue

        roi = frame[:, -7:]  # Extract rightmost 7 pixels (all rows in color frame)

        # Check both templates
        match1, _, y1, = match_template(roi, template1)
        match2, _, y2 = match_template(roi, template2)

        # Process potential matches
        save_match = False
        match_y = None

        if match1:
            if last_match_y is None or abs(y1 - last_match_y) > 3:
                save_match = True
                match_y = y1
        
        if match2 and not save_match:  # Only check template2 if template1 didn't match
            if last_match_y is None or abs(y2 - last_match_y) > 3:
                save_match = True
                match_y = y2

        if save_match:
            timestamp = round(frame_number / fps, 3)
            timestamps.append(timestamp)
            last_match_y = match_y

            print(f"Match found at {timestamp:.3f} seconds (y={match_y})")

        frame_number += 1

    cap.release()
    return timestamps

def find_name(image_path, template_path):

    img = cv2.imread(image_path)
    template = cv2.imread(template_path)  # Read in color
    roi = img[:, -22:]
    match, _, y = match_template(roi, template)

    if not match:
        return False, None

    x = 1665 #spare margin for long names

    name_subsection = img[y:y+48, x:x+148]

    reader = easyocr.Reader(['en', 'ja'])
    
    # Read the image
    results = reader.readtext(name_subsection)
    
    # Returns list of (bounding box, text, confidence)
    for detection in results:
        print(f"Text: {detection[1]}, Confidence: {detection[2]}")
    
    if detection[2] < 0.4:
        return False, None

    return True, detection[1]