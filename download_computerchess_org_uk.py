#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and filter super-grandmaster-level chess games from ComputerChess.org.uk.
Filters games where both players have ratings >= 3300 (super-grandmaster level).
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

MIN_RATING = 3300

import uuid
import py7zr
from collections import defaultdict


def count_games_in_pgn_fast(input_file):
    """
    Count games by scanning for Result tags, which is much faster than parsing.
    Each game ends with a Result tag.
    """
    count = 0
    with open(input_file, 'r') as f:
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


def process_game_chunk(args):
    """
    Process a chunk of games for parallel filtering.
    
    Args:
        args: Tuple of (input_file, output_dir, start_idx, end_idx, min_rating, offset, kept_counter, processed_counter, lock)
    """
    input_file, output_dir, start_idx, end_idx, min_rating, offset, kept_counter, processed_counter, lock = args
    
    local_kept = 0
    local_processed = 0
    
    with open(input_file, 'r') as fh:
        # Skip to the starting position
        for _ in range(start_idx):
            chess.pgn.skip_game(fh)
        
        # Process games in this chunk
        for idx in range(start_idx, end_idx):
            game = chess.pgn.read_game(fh)
            if not game:
                break
            
            local_processed += 1
            
            try:
                if int(game.headers['WhiteElo']) >= min_rating and int(game.headers['BlackElo']) >= min_rating:
                    # Calculate unique filename based on total kept games
                    with lock:
                        file_idx = kept_counter.value
                        kept_counter.value += 1
                    
                    output_file = os.path.join(output_dir, f'{3000000 + offset + file_idx}.pgn')
                    with open(output_file, 'w') as game_fh:
                        print(game, file=game_fh, end='\n\n')
                    local_kept += 1
            except (KeyError, ValueError) as e:
                # Skip games without valid ratings
                pass
            
            # Update shared counters periodically
            if local_processed % 100 == 0:
                with lock:
                    processed_counter.value += 100
                    if processed_counter.value % 10000 == 0:
                        print(f"Processed {processed_counter.value:,} games across all processes")
                local_processed = 0
    
    # Final update
    with lock:
        processed_counter.value += local_processed
    
    return local_kept


def filter_games_by_rating_and_time_control_parallel(input_file, output_directory, min_rating=3300, offset=0, num_processes=None):
    """Filter games where both players have rating >= min_rating using parallel processing."""
    
    if num_processes is None:
        num_processes = min(cpu_count(), 100)
    
    print(f"Filtering games with both players rated >= {min_rating} using {num_processes} processes...")
    
    # For small files or single process, use original function
    if num_processes == 1:
        return filter_games_by_rating_and_time_control(input_file, output_directory, min_rating, offset)
    
    # Setup shared counters
    manager = Manager()
    kept_counter = manager.Value('i', 0)
    processed_counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    # Divide work among processes
    total_games = 2140000  # Approximate number of games
    games_per_process = total_games // num_processes
    chunks = []
    
    for i in range(num_processes):
        start_idx = i * games_per_process
        if i == num_processes - 1:
            end_idx = total_games
        else:
            end_idx = (i + 1) * games_per_process
        
        chunks.append((input_file, output_directory, start_idx, end_idx, min_rating, offset, 
                      kept_counter, processed_counter, lock))
    
    # Process chunks in parallel
    start_time = time.time()
    
    with Pool(num_processes) as pool:
        results = pool.map(process_game_chunk, chunks)
    
    # Calculate totals
    total_kept = sum(results)
    total_processed = processed_counter.value
    elapsed_time = time.time() - start_time
    
    print(f"\nFiltering complete!")
    print(f"Total games processed: {total_processed:,}")
    print(f"Games kept (both players >= {min_rating}): {total_kept:,}")
    if total_processed > 0:
        print(f"Percentage kept: {total_kept/total_processed*100:.1f}%")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Processing speed: {total_processed/elapsed_time:.2f} games/second")
    
    return total_kept, total_processed


def filter_games_by_rating_and_time_control(input_file, output_directory, min_rating=3300, offset=0):
    """Filter games where both players have rating >= min_rating."""
    
    games_processed = 0
    games_kept = 0
    games_skipped = 0

    with open(input_file, 'r') as fh:
        if offset > 0:
            while games_skipped < offset:
                chess.pgn.skip_game(fh)
                games_skipped += 1

        current_game = ""
        in_game = False
        
        print(f"Filtering games with both players rated >= {min_rating}...")
        
        i = 0 + offset

        while True:
            game = chess.pgn.read_game(fh)
            if not game:
                continue
            games_processed += 1
            try:
                if int(game.headers['WhiteElo']) >= min_rating and int(game.headers['BlackElo']) >= min_rating:
                    output_file = os.path.join(output_directory, f'{3000000 + i}.pgn')
                    with open(output_file, 'w') as game_fh:
                        print(game, file=game_fh, end='\n\n')
                    games_kept += 1
                    i += 1
            except KeyError as k:
                print(k)
                continue

            if games_kept % 1000 == 0 and games_kept > 0:
                print(f"Kept {games_kept} games out of {games_processed} games")
            
            if games_processed % 1000 == 0 and games_processed > 0:
                print(f"Processed {games_processed} games out of approximately 2.1MM games")
            
            # There are approximately 2.14MM games here
            if games_processed >= 2140000:
                break

    
    print(f"\nFiltering complete!")
    print(f"Total games processed: {games_processed:,}")
    print(f"Games kept (both players >= {min_rating}): {games_kept:,}")
    print(f"Percentage kept: {games_kept/games_processed*100:.1f}%")
    
    return games_kept, games_processed


def main():
    parser = argparse.ArgumentParser(description="Download and filter grandmaster-level games from ComputerChess.org.uk")
    parser.add_argument('--offset', type=int, default=0, help='Skip the processing of the first N games')
    parser.add_argument('--min-rating', type=int, default=3300, help='Minimum rating for both players (default: 3300)')
    parser.add_argument('--output-dir', default='games_training_data/reformatted', help='Output directory (default: games_training_data/reformatted)')
    parser.add_argument('--skip-download', action='store_true', help='Skip download and only filter existing files')
    parser.add_argument('--output-dir-downloads', default='games_training_data/CCRL_computerchess_org/', help='Output directory to store LiChess Databases (default: games_training_data)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use for parallel filtering (default: number of CPU cores, max 100)')

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
    
    MIN_RATING = args.min_rating or 3300

    # Create output directory for filtered games
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all .pgn files in the output directory

    fName_count = 0
    for filename in sorted(os.listdir(args.output_dir_downloads)):
        if filename.endswith('.pgn'):
            input_path_for_extraction = os.path.join(args.output_dir_downloads, filename)
            input_path_for_parsing = input_path_for_extraction
            output_directory = os.path.join(args.output_dir)
            print(f"\nProcessing {filename}...")
            kept, processed = filter_games_by_rating_and_time_control_parallel(input_path_for_parsing, output_directory, args.min_rating, args.offset, args.processes)
            
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