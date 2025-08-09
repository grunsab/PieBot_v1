"""
Common chess openings for testing MCTS implementations.
Each opening consists of 4 moves (8 half-moves) in UCI notation.
"""

CHESS_OPENINGS = [
    {
        "name": "Italian Game",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1g1", "g8f6"]
    },
    {
        "name": "Ruy Lopez",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
    },
    {
        "name": "Queen's Gambit",
        "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"]
    },
    {
        "name": "Sicilian Defense - Dragon Variation",
        "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"]
    },
    {
        "name": "French Defense",
        "moves": ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5"]
    },
    {
        "name": "King's Indian Defense",
        "moves": ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"]
    },
    {
        "name": "English Opening",
        "moves": ["c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5", "c4d5", "f6d5"]
    },
    {
        "name": "Caro-Kann Defense",
        "moves": ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"]
    },
    {
        "name": "Scotch Game",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4", "f8c5"]
    },
    {
        "name": "Queen's Indian Defense",
        "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8b7"]
    },
    {
        "name": "Nimzo-Indian Defense",
        "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8"]
    },
    {
        "name": "Pirc Defense",
        "moves": ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "f2f4", "f8g7"]
    },
    {
        "name": "Alekhine's Defense",
        "moves": ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6", "g1f3", "c8g4"]
    },
    {
        "name": "Benoni Defense",
        "moves": ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5"]
    },
    {
        "name": "Dutch Defense",
        "moves": ["d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "e7e6", "g1f3", "f8e7"]
    },
    {
        "name": "Scandinavian Defense",
        "moves": ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "g8f6"]
    },
    {
        "name": "Vienna Game",
        "moves": ["e2e4", "e7e5", "b1c3", "g8f6", "f2f4", "d7d5", "f4e5", "f6e4"]
    },
    {
        "name": "King's Gambit",
        "moves": ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g7g5", "h2h4", "g5g4"]
    },
    {
        "name": "London System",
        "moves": ["d2d4", "d7d5", "g1f3", "g8f6", "c1f4", "c7c5", "e2e3", "b8c6"]
    },
    {
        "name": "Catalan Opening",
        "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "f8e7"]
    }
]

def get_opening(index):
    """
    Get an opening by index, cycling through the list if necessary.
    
    Args:
        index: The index of the opening to retrieve
        
    Returns:
        Dictionary containing the opening name and moves
    """
    return CHESS_OPENINGS[index % len(CHESS_OPENINGS)]

def get_total_openings():
    """
    Get the total number of openings available.
    
    Returns:
        Integer count of openings
    """
    return len(CHESS_OPENINGS)