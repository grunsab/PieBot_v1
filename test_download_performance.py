#!/usr/bin/env python3
"""
Test script to compare performance of single vs multi-process game filtering.
"""

import time
import os
import tempfile
import shutil
from download_lichess_games import (
    filter_games_by_rating_and_time_control,
    filter_games_by_rating_and_time_control_parallel
)

def test_performance():
    """Compare single-process vs multi-process performance"""
    
    # Check if we have a test file
    test_files = [
        'games_training_data/LiChessData/lichess_db_standard_rated_2024-11.pgn.zst',
        'games_training_data/LiChessData/lichess_db_standard_rated_2024-10.pgn.zst',
    ]
    
    test_file = None
    for f in test_files:
        if os.path.exists(f):
            test_file = f
            break
    
    if not test_file:
        print("No test file found. Please download a Lichess database file first.")
        print("Run: python3 download_lichess_games.py --months 1")
        return
    
    print(f"Using test file: {test_file}")
    
    # Test parameters
    max_games = 1000  # Small number for quick testing
    min_rating = 1500  # Higher rating to reduce matches
    
    # Test single-process mode
    print("\n" + "="*60)
    print("Testing SINGLE-PROCESS mode...")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        single_output = os.path.join(temp_dir, "single")
        
        start = time.time()
        kept, processed = filter_games_by_rating_and_time_control(
            test_file, single_output, 
            min_rating=min_rating, 
            max_games=max_games
        )
        single_time = time.time() - start
        
        print(f"Time: {single_time:.2f}s")
        print(f"Games processed: {processed}")
        print(f"Games kept: {kept}")
        print(f"Speed: {processed/single_time:.1f} games/sec")
    
    # Test multi-process mode with different worker counts
    for num_workers in [2, 4, 8]:
        print("\n" + "="*60)
        print(f"Testing MULTI-PROCESS mode with {num_workers} workers...")
        print("="*60)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            multi_output = os.path.join(temp_dir, "multi")
            
            start = time.time()
            kept, processed = filter_games_by_rating_and_time_control_parallel(
                test_file, multi_output,
                min_rating=min_rating,
                max_games=max_games,
                num_processes=num_workers
            )
            multi_time = time.time() - start
            
            print(f"Time: {multi_time:.2f}s")
            print(f"Games processed: {processed}")
            print(f"Games kept: {kept}")
            print(f"Speed: {processed/multi_time:.1f} games/sec")
            
            if single_time > 0:
                speedup = single_time / multi_time
                print(f"Speedup vs single-process: {speedup:.2f}x")
                if speedup < 1:
                    print(f"  WARNING: {num_workers} workers is {1/speedup:.2f}x SLOWER than single-process!")

if __name__ == "__main__":
    test_performance()