from collections import Counter

def most_frequent_string(string_list):
  """
    Finds the most frequent string in a list.

    Args:
      string_list: A list of strings.

    Returns:
      The most frequent string in the list.
      If the list is empty, returns None.
      If there are multiple strings with the same highest frequency, returns the first one encountered.
  """
  if not string_list:
    return None

  string_counts = Counter(string_list)
  most_common_string = string_counts.most_common(1)[0][0]
  return most_common_string

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def match_ocr_name(actual_players, ocr_result):
    """
    Match OCR-detected names with the closest actual player names.
    
    Args:
        actual_players (list): List of actual player names
        ocr_result (str): OCR-detected name string
        
    Returns:
        tuple: (success, player_name) indicating whether a match was found and the matched player name
    """
    matched_results = {}
    
    ocr_name_lower = ocr_result.lower()
    
    # Calculate distances to all actual player names
    distances = []
    for i, player_name in enumerate(actual_players):
        player_lower = player_name.lower()
        dist = levenshtein_distance(ocr_name_lower, player_lower)
        distances.append((dist, i))
    
    # Find the closest match
    if distances:
        min_distance, closest_index = min(distances, key=lambda x: x[0])
        closest_player = actual_players[closest_index]

        if min_distance > 4:
            return False, None
        
        return True, closest_player
    else:
        return False, None