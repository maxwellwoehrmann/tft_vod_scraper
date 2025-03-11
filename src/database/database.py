import os
import json
from typing import List, Dict, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv
from datetime import datetime
from ..utils import logger

# Global database connection
_db_client = None
_db = None

def connect_to_database():
    """Connect to MongoDB database."""
    global _db_client, _db
    log = logger.get_logger(__name__)
    
    # Return existing connection if available
    if _db is not None:
        return _db
    
    # Load environment variables
    load_dotenv()
    
    # Get connection string from environment variable
    connection_string = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    
    try:
        # Connect to MongoDB
        _db_client = MongoClient(connection_string)
        _db = _db_client.tft_augments
        
        # Create indexes for better query performance
        setup_indexes()
        
        log.info("Connected to MongoDB successfully")
        return _db
    except Exception as e:
        log.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
        raise

def setup_indexes():
    """Set up database indexes for better query performance."""
    db = connect_to_database()
    log = logger.get_logger(__name__)
    
    # Index for game_id (unique)
    db.games.create_index([("game_id", ASCENDING)], unique=True)
    
    # Index for match_id
    db.games.create_index([("match_id", ASCENDING)])
    
    # Index for complete_data flag (frequently used in filtering)
    db.games.create_index([("complete_data", ASCENDING)])
    
    log.info("Database indexes set up")

def add_game(game_data: Dict) -> str:
    """Add a new game to the database."""
    db = connect_to_database()
    log = logger.get_logger(__name__)
    
    try:
        # Add timestamp
        if 'timestamp' not in game_data:
            game_data['timestamp'] = datetime.now()
        
        # Calculate if data is complete (all 8 players analyzed)
        players_analyzed = len(game_data.get('players', []))
        game_data['players_analyzed'] = players_analyzed
        game_data['complete_data'] = players_analyzed == 8
        
        # Add or update game
        result = db.games.update_one(
            {"game_id": game_data["game_id"]},
            {"$set": game_data},
            upsert=True
        )
        
        if result.upserted_id:
            log.info(f"Added new game: {game_data['game_id']}")
        else:
            log.info(f"Updated existing game: {game_data['game_id']}")
            
        return game_data['game_id']
    except Exception as e:
        log.error(f"Failed to add game {game_data.get('game_id', 'unknown')}: {e}", exc_info=True)
        raise

def get_game(game_id: str) -> Optional[Dict]:
    """Retrieve a game by its game_id."""
    db = connect_to_database()
    log = logger.get_logger(__name__)
    
    try:
        return db.games.find_one({"game_id": game_id})
    except Exception as e:
        log.error(f"Failed to retrieve game {game_id}: {e}", exc_info=True)
        raise

def get_known_game_ids() -> List[str]:
    """Get a list of all game IDs in the database."""
    db = connect_to_database()
    log = logger.get_logger(__name__)
    
    try:
        return [doc["game_id"] for doc in db.games.find({}, {"game_id": 1})]
    except Exception as e:
        log.error(f"Failed to retrieve known game IDs: {e}", exc_info=True)
        raise

def filter_known_games(vods: List[Dict]) -> List[Dict]:
    """Filter out VODs that are already in the database."""
    log = logger.get_logger(__name__)
    
    try:
        known_game_ids = set(get_known_game_ids())
        filtered_vods = [vod for vod in vods if vod['game_id'] not in known_game_ids]
        log.debug(f"Filtered {len(vods) - len(filtered_vods)} known games out of {len(vods)} total")
        return filtered_vods
    except Exception as e:
        log.error(f"Failed to filter known games: {e}", exc_info=True)
        raise

def save_results(placements: List[Dict], augment_data: Dict, match_id: str) -> None:
    """Save match results to the database."""
    db = connect_to_database()
    log = logger.get_logger(__name__)
    
    try:
        complete_data = True
        # Get the game by match_id
        game = db.games.find_one({"match_id": match_id})
        if not game:
            log.warning(f"Game with match_id {match_id} not found when saving results")
            return
        
        analyzed = 0
        # Construct player data
        players = []
        for placement in placements:
            player_name = placement['name'].lower()
            
            # Skip if this player's augments weren't detected
            if player_name not in augment_data:
                log.debug(f"Missing augment data for player: {player_name}")
                complete_data = False
                continue
            elif len(augment_data[player_name]) != 3:
                log.debug(f"Incomplete augment data for player: {player_name} ({len(augment_data[player_name])}/3 augments)")
                complete_data = False
            else:
                analyzed += 1
            
            player_augments = augment_data[player_name]
            augments = []
            
            # Format augments data
            for position, augment_name in player_augments.items():
                augments.append({
                    "position": int(position),
                    "name": augment_name
                })
            
            players.append({
                "name": player_name,
                "placement": placement['placement'],
                "level": placement.get('level'),
                "last_round": placement.get('last_round'),
                "augments": augments
            })
        
        # Update game with player data
        db.games.update_one(
            {"match_id": match_id},
            {
                "$set": {
                    "players": players,
                    "players_analyzed": analyzed,
                    "complete_data": complete_data
                }
            }
        )
        
        log.info(f"Saved results for match {match_id} with {len(players)} players ({analyzed} fully analyzed)")
        log.debug(f"Data completeness: {complete_data}")
    except Exception as e:
        log.error(f"Failed to save results for match {match_id}: {e}", exc_info=True)
        raise

def get_augment_performance(min_games=5, complete_games_only=True):
    """
    Get augment performance statistics by stage.
    
    Args:
        min_games: Minimum number of games an augment must appear in
        complete_games_only: Whether to only include complete games (all 8 players)
        
    Returns:
        Dictionary with augment stats by position
    """
    db = connect_to_database()
    log = logger.get_logger(__name__)
    
    try:
        # Build query conditions
        match_condition = {}
        if complete_games_only:
            match_condition["complete_data"] = True
        
        log.info(f"Analyzing augment performance (min_games={min_games}, complete_only={complete_games_only})")
        
        # Create aggregation pipeline
        pipeline = [
            # Match condition for complete games if specified
            {"$match": match_condition},
            
            # Unwind arrays to analyze at augment level
            {"$unwind": "$players"},
            {"$unwind": "$players.augments"},
            
            # Group by augment name and position
            {"$group": {
                "_id": {
                    "name": "$players.augments.name",
                    "position": "$players.augments.position"
                },
                "total_games": {"$sum": 1},
                "avg_placement": {"$avg": "$players.placement"},
                "top4_count": {"$sum": {"$cond": [{"$lte": ["$players.placement", 4]}, 1, 0]}},
                "win_count": {"$sum": {"$cond": [{"$eq": ["$players.placement", 1]}, 1, 0]}}
            }},
            
            # Filter augments with sufficient sample size
            {"$match": {"total_games": {"$gte": min_games}}},
            
            # Add calculated fields
            {"$project": {
                "_id": 0,
                "name": "$_id.name",
                "position": "$_id.position",
                "total_games": 1,
                "avg_placement": {"$round": ["$avg_placement", 2]},
                "top4_rate": {"$round": [{"$divide": ["$top4_count", "$total_games"]}, 3]},
                "win_rate": {"$round": [{"$divide": ["$win_count", "$total_games"]}, 3]}
            }},
            
            # Sort by position, then by average placement
            {"$sort": {"position": 1, "avg_placement": 1}}
        ]
        
        # Execute the aggregation
        results = list(db.games.aggregate(pipeline))
        log.info(f"Found performance data for {len(results)} augments across all positions")
        
        # Organize results by position for easier consumption
        performance_by_position = {0: [], 1: [], 2: []}
        
        for augment in results:
            position = augment["position"]
            if position in performance_by_position:
                performance_by_position[position].append(augment)
        
        # Log summary stats
        for position, augments in performance_by_position.items():
            log.info(f"Position {position}: {len(augments)} augments with sufficient data")
        
        return performance_by_position
    
    except Exception as e:
        log.error(f"Failed to retrieve augment performance: {e}", exc_info=True)
        raise

def close_connection():
    """Close the database connection."""
    global _db_client, _db
    log = logger.get_logger(__name__)
    
    if _db_client is not None:
        _db_client.close()
        _db_client = None
        _db = None
        log.info("Closed MongoDB connection")