#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and filter chess games from Lichess database.
Filters games where both players have ratings >= 750 (beginner level).
"""

import requests
import os
import sys
import io
import gzip
import shutil
from datetime import datetime
import re
import argparse
from tqdm import tqdm
import time
import chess.pgn 
import sys
import argparse
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool, Manager, cpu_count
import time
from functools import partial
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

MIN_RATING = 750
MAX_GAMES_TO_COLLECT = 70000000


import uuid
import zstandard as zstd



def count_games_in_pgn_fast(input_file):
    """
    Count games by scanning for Result tags, which is much faster than parsing.
    Each game ends with a Result tag.
    """
    count = 0
    with zstd.open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('[Result '):
                count += 1
    return count

    
def count_games_parallel(args):
    """
    Count games in a single file for parallel processing.
    """
    pgn_file, input_dir = args
    input_path = os.path.join(input_dir, pgn_file)
    game_count = count_games_in_pgn_fast(input_path)
    return (pgn_file, input_path, game_count)


# Fix Windows console encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Lichess database URLs
LICHESS_DB_URL = "https://database.lichess.org/"

def get_available_databases():
    """Fetch list of available Lichess databases."""
    response = requests.get(LICHESS_DB_URL)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch database list: {response.status_code}")
    
    # Parse HTML to find .pgn.zst files
    pattern = r'href="(standard/lichess_db_standard_rated_\d{4}-\d{2}\.pgn\.zst)"'
    matches = re.findall(pattern, response.text)
    
    return sorted(matches, reverse=True)  # Most recent first

def download_file(url_filename_pairs, chunk_size=8192):
    """Download a file with progress bar."""
    url = url_filename_pairs[0]
    filepath = url_filename_pairs[1]
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
            for data in response.iter_content(chunk_size):
                pbar.update(len(data))
                f.write(data)

def parallel_download_mp(url_filename_pairs):
    """Downloads multiple files in parallel using multiprocessing.Pool."""
    num_processes = cpu_count() - 1 if cpu_count() > 1 else 1 # Use one less than available CPUs
    num_processes = max(num_processes, len(url_filename_pairs))
    with Pool(processes=num_processes) as pool:
        pool.map(download_file, url_filename_pairs)


def extract_and_verify_rating(pgn_headers):
    """Extract white and black ratings from PGN headers."""
    white_rating = None
    black_rating = None
    
    for line in pgn_headers.split('\n'):
        if line.startswith('[WhiteElo'):
            match = re.search(r'"(\d+)"', line)
            if match:
                white_rating = int(match.group(1))
        elif line.startswith('[BlackElo'):
            match = re.search(r'"(\d+)"', line)
            if match:
                black_rating = int(match.group(1))
    
    return (white_rating >= MIN_RATING and black_rating >= MIN_RATING)

def extract_and_verify_time_controls(pgn_headers, min_seconds=180):
    """Extract white and black ratings from PGN headers."""

    time_control = None

    time_control_match = re.search(r'\[TimeControl "(\d+)\+(\d+)"\]', pgn_headers)

    if time_control_match:
        initial_time, increment = time_control_match.groups()
    else:
        return False

    return int(initial_time) >= 180




def filter_games_by_rating_and_time_control(input_file, output_dir, min_rating=750, min_seconds=180):
    """Filter games where both players have rating >= min_rating and save each to individual files."""
    
    games_processed = 0
    games_kept = 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Filtering games with both players rated >= {min_rating}...")
    print(f"Saving individual games to: {output_dir}")
    
    # Open compressed input file with text mode streaming
    with open(input_file, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            # Use text wrapper for proper line handling
            text_reader = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')
            
            current_game = ""
            in_game = False
            
            for line in text_reader:
                line = line.rstrip('\n\r')
                
                if line.startswith('[Event'):
                    # Start of new game
                    if current_game and in_game:
                        # Process previous game
                        rating_check = extract_and_verify_rating(current_game)
                        time_control_check = extract_and_verify_time_controls(current_game)
                        if rating_check and time_control_check:
                            # Save each game to its own file with lichess prefix
                            game_filename = f"lichess_{games_kept:06d}.pgn"
                            game_filepath = os.path.join(output_dir, game_filename)
                            with open(game_filepath, 'w', encoding='utf-8') as game_file:
                                game_file.write(current_game + '\n\n')
                            games_kept += 1
                        games_processed += 1
                        
                        if games_processed % 10000 == 0:
                            print(f"Processed: {games_processed:,} games, Kept: {games_kept:,} games ({games_kept/games_processed*100:.1f}%)")
                        
                        # Check if we've collected enough games
                        if games_kept >= MAX_GAMES_TO_COLLECT:
                            break
                    
                    current_game = line + '\n'
                    in_game = True
                elif in_game:
                    current_game += line + '\n'
            
            # Process last game
            if current_game and in_game:
                rating_check = extract_and_verify_rating(current_game)
                time_control_check = extract_and_verify_time_controls(current_game)
                if rating_check and time_control_check:
                    # Save each game to its own file with lichess prefix
                    game_filename = f"lichess_{games_kept:06d}.pgn"
                    game_filepath = os.path.join(output_dir, game_filename)
                    with open(game_filepath, 'w', encoding='utf-8') as game_file:
                        game_file.write(current_game + '\n\n')
                    games_kept += 1
                games_processed += 1



    
    print(f"\nFiltering complete!")
    print(f"Total games processed: {games_processed:,}")
    print(f"Games kept (both players >= {min_rating}): {games_kept:,}")
    print(f"Percentage kept: {games_kept/games_processed*100:.1f}%")
    
    return games_kept, games_processed


def main():
    parser = argparse.ArgumentParser(description="Download and filter games from Lichess")
    parser.add_argument('--months', type=int, default=1, help='Number of months to download (default: 1)')
    parser.add_argument('--min-rating', type=int, default=750, help='Minimum rating for both players (default: 750)')
    parser.add_argument('--output-dir', default='games_training_data/reformatted_lichess', help='Output directory (default: games_training_data/reformatted_lichess)')
    parser.add_argument('--skip-download', action='store_true', help='Skip download and only filter existing files')
    parser.add_argument('--output-dir-downloads', default='games_training_data/LiChessData/', help='Output directory to store LiChess Databases (default: games_training_data)')

    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir_downloads, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.skip_download:
        # Get available databases
        print("Fetching available Lichess databases...")
        databases = get_available_databases()
        
        if not databases:
            print("No databases found!")
            return
        
        # Download requested number of months
        databases_to_download = databases[:args.months]
        
        print(f"\nFound {len(databases)} databases. Will download {len(databases_to_download)}:")
        for db in databases_to_download:
            print(f"  - {db}")
        
        # Download files

        url_paths_and_fnames = []

        for db_path in databases_to_download:
            filename = os.path.basename(db_path)
            filepath = os.path.join(args.output_dir_downloads, filename)
            url = LICHESS_DB_URL + db_path
            url_paths_and_fnames.append((url, filepath))

        parallel_download_mp(url_paths_and_fnames)
    
    # Filter downloaded files
    print("\n" + "="*50)
    print("Starting filtering process...")
    print("="*50)
    
    
    total_kept = 0
    total_processed = 0
    
    MIN_RATING = args.min_rating or 750

    # Process all .pgn.zst files in the output directory
    print(f"\nLooking for .pgn.zst files in: {args.output_dir_downloads}")
    
    zst_files = [f for f in os.listdir(args.output_dir_downloads) if f.endswith('.pgn.zst')]
    print(f"Found {len(zst_files)} compressed files to process")
    
    fName_count = 0
    for filename in sorted(zst_files):
        input_path_for_extraction = os.path.join(args.output_dir_downloads, filename)
        input_path_for_parsing = input_path_for_extraction
        # Use output_dir directly as the directory for individual game files
        output_directory = args.output_dir
        print(f"\nProcessing {filename}...")
        print(f"  Input: {input_path_for_parsing}")
        print(f"  Output directory: {output_directory}")
        kept, processed = filter_games_by_rating_and_time_control(input_path_for_parsing, output_directory, min_rating=MIN_RATING)
        
        total_kept += kept
        total_processed += processed
        fName_count += 1
    
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Total games processed: {total_processed:,}")
    print(f"Total games kept: {total_kept:,}")
    if total_processed > 0:
        print(f"Overall percentage: {total_kept/total_processed*100:.1f}%")
    print(f"\nFiltered games saved in: {args.output_dir}/")

if __name__ == "__main__":
    main()