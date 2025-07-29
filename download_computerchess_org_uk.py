#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and filter grandmaster-level chess games from Lichess database.
Filters games where both players have ratings >= 2850 (grandmaster level).
"""

import requests
import os
import sys
import gzip
import shutil
from datetime import datetime
import re
import argparse
from tqdm import tqdm
import time
from extract_zst import extract_zst
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

MIN_RATING = 3000

import uuid
import zstandard as zstd
import py7zr



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



def filter_games_by_rating_and_time_control(input_file, output_directory, min_rating=3000, min_seconds=180):
    """Filter games where both players have rating >= min_rating."""
    
    games_processed = 0
    games_kept = 0
    
    # Open compressed input file
    with open(input_file, 'r') as fh:
        current_game = ""
        in_game = False
        
        print(f"Filtering games with both players rated >= {min_rating}...")
        
        i = 0

        while True:
            game = chess.pgn.read_game(fh)
            if not game:
                break
            games_processed += 1
            if game.headers['WhiteElo'] >= min_rating and game.headers['BlackElo'] >= min_rating:
                output_file = os.path.join(output_directory, f'{3000000 + i}.pgn')
                with open(output_file, 'w') as game_fh:
                    print(game, file=game_fh, end='\n\n')
                games_kept += 1
                i += 1

            if games_kept % 1000 == 0:
                print("Kept {game_kept} games out of {games_processed} games")
    
    print(f"\nFiltering complete!")
    print(f"Total games processed: {games_processed:,}")
    print(f"Games kept (both players >= {min_rating}): {games_kept:,}")
    print(f"Percentage kept: {games_kept/games_processed*100:.1f}%")
    
    return games_kept, games_processed


def main():
    parser = argparse.ArgumentParser(description="Download and filter grandmaster-level games from Lichess")
    parser.add_argument('--months', type=int, default=1, help='Number of months to download (default: 1)')
    parser.add_argument('--min-rating', type=int, default=2850, help='Minimum rating for both players (default: 2850)')
    parser.add_argument('--output-dir', default='games_training_data/reformatted', help='Output directory (default: games_training_data/reformatted)')
    parser.add_argument('--skip-download', action='store_true', help='Skip download and only filter existing files')
    parser.add_argument('--output-dir-downloads', default='games_training_data/CCRL_computerchess_org/', help='Output directory to store LiChess Databases (default: games_training_data)')

    args = parser.parse_args()
    
    # Create output directory downloads 
    os.makedirs(args.output_dir_downloads, exist_ok=True)
    
    if not args.skip_download:
        # Get available databases
        print("Fetching CCRL dataset...")
        
        url_path = "https://computerchess.org.uk/ccrl/4040/CCRL-4040.[2140066].pgn.7z"
        fileName = "CCRL-4040.[2140066].pgn.7z"
        url_paths_and_fnames = (url_path, fileName)
        download_file(url_paths_and_fnames)

        archive_path = os.path.join(fileName)
        output_directory = os.path.join(args.output_dir_downloads)

    
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
        # For password-protected archives, uncomment the line below and provide the password
        # with py7zr.SevenZipFile(archive_path, mode='r', password=password) as z:
            z.extractall(path=output_directory)
            print(f"Archive '{archive_path}' successfully extracted to '{output_directory}'")


    
    
    
    total_kept = 0
    total_processed = 0
    
    MIN_RATING = args.min_rating or 2850

    # Process all .pgn files in the output directory

    fName_count = 0
    for filename in sorted(os.listdir(args.output_dir_downloads)):
        if filename.endswith('.pgn'):
            input_path_for_extraction = os.path.join(args.output_dir_downloads, filename)
            input_path_for_parsing = input_path_for_extraction
            output_directory = os.path.join(args.output_dir)
            print(f"\nProcessing {filename}...")
            kept, processed = filter_games_by_rating_and_time_control(input_path_for_parsing, output_directory)
            
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