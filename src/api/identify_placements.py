import requests
import json
import os
from dotenv import load_dotenv
from ..utils import logger

# Load API key from environment variables
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")

def get_tft_match_placements(match_id):
    """
    Get placements from a TFT match using the Riot API
    
    Args:
        match_id (str): Match ID (e.g., 'EUW1_7324928302')
    
    Returns:
        list: List of players with their placements or None if error
    """
    log = logger.get_logger(__name__)
    
    # Extract region code from match ID
    try:
        region_code = match_id.split('_')[0].lower()
        log.debug(f"Extracted region code: {region_code} from match ID: {match_id}")
    except Exception as e:
        log.error(f"Failed to extract region from match_id {match_id}: {e}")
        return None
    
    # Map region code to routing value
    routing_values = {
        # Europe
        'euw1': 'europe',
        'eun1': 'europe',
        'tr1': 'europe',
        'ru': 'europe',
        # Americas
        'na1': 'americas',
        'br1': 'americas',
        'la1': 'americas',
        'la2': 'americas',
        # Asia
        'kr': 'asia',
        'jp1': 'asia',
        # Sea
        'oc1': 'sea',
        'ph2': 'sea',
        'sg2': 'sea',
        'th2': 'sea',
        'tw2': 'sea',
        'vn2': 'sea'
    }
    
    routing_value = routing_values.get(region_code, 'europe')
    log.debug(f"Mapped {region_code} to routing value: {routing_value}")
    
    # Check if API key is available
    if not RIOT_API_KEY:
        log.error("RIOT_API_KEY environment variable not set")
        return None

    # Construct the URL for the TFT match endpoint
    url = f"https://{routing_value}.api.riotgames.com/tft/match/v1/matches/{match_id}?api_key={RIOT_API_KEY}"
    log.debug(f"Making API request to: {url.replace(RIOT_API_KEY, 'API_KEY_REDACTED')}")

    try:
        response = requests.get(url)
        log.debug(f"API response status code: {response.status_code}")
        
        if response.status_code == 200:
            match_data = response.json()
            
            # Extract placements
            placements = []
            participant_count = len(match_data['info']['participants'])
            log.info(f"Retrieved match data with {participant_count} participants")
            
            for participant in match_data['info']['participants']:
                placements.append({
                    'name': participant['riotIdGameName'],
                    'tagline': participant.get('riotIdTagline', ''),
                    'placement': participant['placement'],
                    'level': participant['level'],
                    'last_round': participant['last_round']
                })
            
            # Sort by placement
            placements = sorted(placements, key=lambda x: x['placement'])
            log.info(f"Processed {len(placements)} player placements for match {match_id}")
            
            return placements
        elif response.status_code == 404:
            log.warning(f"Match not found: {match_id}")
            return None
        elif response.status_code == 401 or response.status_code == 403:
            log.error(f"API authentication error: {response.status_code}")
            log.debug(f"API response: {response.text}")
            return None
        else:
            log.error(f"API request failed with status code: {response.status_code}")
            log.debug(f"API response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        log.error(f"Request error for match {match_id}: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error for match {match_id}: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Configure logging
    logger.configure_logger(log_level=logger.DEBUG)
    log = logger.get_logger(__name__)
    
    # Get match ID
    match_id = "EUW1_7324928302"
    log.info(f"Testing placement retrieval for match {match_id}")
    
    # Get placements
    placements = get_tft_match_placements(match_id)
    
    if placements:
        log.info("\nMatch Placements:")
        log.info("=" * 40)
        for player in placements:
            log.info(f"#{player['placement']}: {player['name']}#{player['tagline']} (Level {player['level']}, Last Round {player['last_round']})")
    else:
        log.error("Failed to retrieve match data")