import requests
import json
import os
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")

def get_tft_match_placements(match_id):
    """
    Get placements from a TFT match using the Riot API
    
    Args:
        match_id (str): Match ID (e.g., 'EUW1_7324928302')
    
    Returns:
        list: List of players with their placements
    """
    # Extract region code from match ID
    region_code = match_id.split('_')[0].lower()
    
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

    # Construct the URL for the TFT match endpoint
    url = f"https://{routing_value}.api.riotgames.com/tft/match/v1/matches/{match_id}?api_key={RIOT_API_KEY}"

    try:
        print(url)
        response = requests.get(url)
        
        if response.status_code == 200:
            match_data = response.json()
            
            # Extract placements
            placements = []
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
            return placements
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Get match ID
    match_id = "EUW1_7324928302"
    
    # Get placements
    placements = get_tft_match_placements(match_id)
    
    if placements:
        print("\nMatch Placements:")
        print("=" * 40)
        for player in placements:
            print(f"#{player['placement']}: {player['name']}#{player['tagline']} (Level {player['level']}, Last Round {player['last_round']})")
    else:
        print("Failed to retrieve match data")