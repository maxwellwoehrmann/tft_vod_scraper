from pathlib import Path
from typing import List, Dict

def filter_known_games(vods: List[Dict]) -> List[Dict]:
    """
    Remove VODs that are already in games.csv
    """
    games_file = Path(__file__).parent.parent.parent / 'data' / 'games.csv'
    
    try:
        with open(games_file, 'r') as f:
            known_games = set(int(line.strip()) for line in f if line.strip())
    except FileNotFoundError:
        known_games = set()
    
    return [vod for vod in vods if vod['game_id'] not in known_games]

def add_games_to_csv(vods: List[Dict]) -> None:
   """
   Append new game IDs to games.csv. Each game ID on a new line.
   """
   games_file = Path(__file__).parent.parent.parent / 'data' / 'games.csv'
   
   # Create data directory if it doesn't exist
   games_file.parent.mkdir(exist_ok=True)
   
   # Append game IDs to file
   with open(games_file, 'a') as f:
       for vod in vods:
           f.write(f"{vod['game_id']}\n")