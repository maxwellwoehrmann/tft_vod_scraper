import requests
import json
from typing import List, Dict
from datetime import datetime
import re
from ..utils import logger

def parse_twitch_timestamp(url: str) -> str:
    """
    Convert Twitch timestamp (e.g. 't=1h26m5s') to HH:MM:SS format
    """
    log = logger.get_logger(__name__)
    
    # Extract the timestamp part
    match = re.search(r't=(\d+h)?(\d+m)?(\d+s)?', url)
    if not match:
        log.debug(f"No timestamp found in URL: {url}, using default 00:00:00")
        return "00:00:00"
        
    hours = int(match.group(1)[:-1]) if match.group(1) else 0
    minutes = int(match.group(2)[:-1]) if match.group(2) else 0
    seconds = int(match.group(3)[:-1]) if match.group(3) else 0
    
    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    log.debug(f"Parsed timestamp: {timestamp} from {url}")
    return timestamp

def format_seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    seconds = int(seconds) % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def fetch_recent_vods(amnt, offset) -> List[Dict]:
    """
    Fetch recent TFT VODs and extract relevant info.
    Returns list of dicts with VOD url and game timing info in HH:MM:SS format.
    
    Args:
        amnt: Number of VODs to fetch
        offset: Offset for pagination
        
    Returns:
        List of VOD information dictionaries
    """
    log = logger.get_logger(__name__)
    
    url = f"https://api.metatft.com/tft-vods/latest?placement=1,2&limit={amnt}&offset={offset}"
    log.info(f"Fetching {amnt} VODs with offset {offset}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        log.info(f"Retrieved {len(data)} VODs from API")
        
        vods = []
        for record in data:
            try:
                match_data = json.loads(record["match_data"])
                game_length = round(match_data["info"]["game_length"])
                
                # Get start time and format as HH:MM:SS
                vod_start = parse_twitch_timestamp(record["twitch_vod"])
                # Convert game_length to HH:MM:SS format
                game_duration = format_seconds_to_timestamp(game_length)
                
                # Calculate end time by adding seconds to the parsed start time
                h, m, s = map(int, vod_start.split(':'))
                total_start_seconds = h * 3600 + m * 60 + s
                vod_end = format_seconds_to_timestamp(total_start_seconds + game_length)
                
                players = [
                    player["riot_id"].split('#')[0].lower()
                    for player in match_data["_metatft"]["participant_info"]
                ]
                
                vod_info = {
                    "vod_url": record["twitch_vod"],
                    "game_start": vod_start,
                    "game_finish": vod_end,
                    "game_id": match_data["info"]["gameId"],
                    "match_id": record["match_id"],
                    "players": players,
                    "placement": record["placement"]
                }
                vods.append(vod_info)
                log.debug(f"Processed VOD {vod_info['game_id']} ({game_duration} duration)")
                
            except (KeyError, ValueError, TypeError) as e:
                log.warning(f"Error processing VOD record: {e}", exc_info=True)
                log.debug(f"Problematic record: {record}")
                continue
        
        log.info(f"Successfully processed {len(vods)} VODs")
        return vods
        
    except requests.exceptions.RequestException as e:
        log.error(f"API request failed: {e}", exc_info=True)
        return []
    except Exception as e:
        log.error(f"Unexpected error fetching VODs: {e}", exc_info=True)
        return []