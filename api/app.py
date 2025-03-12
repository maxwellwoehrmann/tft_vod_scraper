from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# MongoDB connection
def get_db_connection():
    connection_string = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    client = MongoClient(connection_string)
    return client.tft_augments

@app.route('/api/augments', methods=['GET'])
def get_augments():
    """Get augment performance data with position breakdown"""
    min_games = request.args.get('min_games', default=5, type=int)
    complete_only = request.args.get('complete_only', default='true', type=str)
    complete_only = complete_only.lower() == 'true'
    search = request.args.get('search', default='', type=str)
    
    # Connect to database
    db = get_db_connection()
    
    # Build aggregation pipeline
    match_condition = {}
    if complete_only:
        match_condition["complete_data"] = True
    
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
    
    # Process the results to create a combined view by augment
    augment_data = {}
    
    for augment in results:
        name = augment["name"]
        position = augment["position"]
        
        # Initialize augment if not seen before
        if name not in augment_data:
            augment_data[name] = {
                "name": name,
                "total_games": 0,
                "avg_placement": 0,
                "position_data": {
                    "0": None,  # 2-1
                    "1": None,  # 3-2
                    "2": None   # 4-2
                }
            }
        
        # Update position-specific data
        position_str = str(position)
        augment_data[name]["position_data"][position_str] = {
            "avg_placement": augment["avg_placement"],
            "total_games": augment["total_games"],
            "top4_rate": augment["top4_rate"],
            "win_rate": augment["win_rate"]
        }
        
        # Add to total games
        augment_data[name]["total_games"] += augment["total_games"]
    
    # Calculate overall average placement across positions
    for name, data in augment_data.items():
        total_weighted_placement = 0
        total_games = 0
        
        for pos, pos_data in data["position_data"].items():
            if pos_data:
                total_weighted_placement += pos_data["avg_placement"] * pos_data["total_games"]
                total_games += pos_data["total_games"]
        
        if total_games > 0:
            data["avg_placement"] = round(total_weighted_placement / total_games, 2)
    
    # Convert to list for easier consumption
    augment_list = list(augment_data.values())
    
    # Apply search filter if provided
    if search:
        search = search.lower()
        augment_list = [a for a in augment_list if search in a["name"].lower()]
    
    # Sort by average placement by default
    augment_list.sort(key=lambda x: x["avg_placement"])
    
    return jsonify(augment_list)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "TFT Augment API is running"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
