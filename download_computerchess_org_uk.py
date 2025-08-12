#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and filter super-grandmaster-level chess games from ComputerChess.org.uk.
Filters games where both players have ratings >= 2500 (significantly beyond human level).
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
# import chess.pgn  # No longer needed - using string-based extraction 
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

MIN_RATING = 2500

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


def count_existing_games_in_directory(directory):
    """
    Count the number of already processed games in the output directory.
    Games are expected to be named with numeric pattern (e.g., 0.pgn, 1.pgn, etc.)
    """
    if not os.path.exists(directory):
        return 0
    
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.pgn'):
            # Check if filename (without .pgn) is a number
            try:
                int(filename[:-4])
                count += 1
            except ValueError:
                # Not a numeric filename, skip
                pass
    
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


def extract_and_verify_rating(pgn_headers, min_rating):
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
    
    if white_rating is None or black_rating is None:
        return False
    return white_rating >= min_rating and black_rating >= min_rating


def process_game_chunk(args):
    """
    Process a chunk of lines for parallel filtering using string-based extraction.
    
    Args:
        args: Tuple of (input_file, output_dir, start_line, end_line, min_rating, offset, kept_counter, processed_counter, lock)
    """
    input_file, output_dir, start_line, end_line, min_rating, offset, kept_counter, processed_counter, lock = args
    
    local_kept = 0
    local_processed = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fh:
        current_game = ""
        in_game = False
        current_line_num = 0
        
        for line in fh:
            current_line_num += 1
            
            # Skip lines before our chunk
            if current_line_num < start_line:
                continue
            
            # Stop after our chunk
            if current_line_num > end_line:
                break
            
            line = line.rstrip('\n\r')
            
            if line.startswith('[Event'):
                # Start of new game
                if current_game and in_game:
                    # Process previous game
                    local_processed += 1
                    if extract_and_verify_rating(current_game, min_rating):
                        # Calculate unique filename based on total kept games
                        with lock:
                            file_idx = kept_counter.value
                            kept_counter.value += 1
                        
                        output_file = os.path.join(output_dir, f'{offset + file_idx}.pgn')
                        with open(output_file, 'w', encoding='utf-8') as game_fh:
                            game_fh.write(current_game + '\n\n')
                        local_kept += 1
                    
                    # Update shared counters periodically
                    if local_processed % 1000 == 0:
                        with lock:
                            processed_counter.value += 1000
                            if processed_counter.value % 20000 == 0:
                                print(f"Processed {processed_counter.value:,} games across all processes")
                        local_processed = 0
                
                current_game = line + '\n'
                in_game = True
            elif in_game:
                current_game += line + '\n'
        
        # Process last game in chunk
        if current_game and in_game:
            local_processed += 1
            if extract_and_verify_rating(current_game, min_rating):
                with lock:
                    file_idx = kept_counter.value
                    kept_counter.value += 1
                
                output_file = os.path.join(output_dir, f'{offset + file_idx}.pgn')
                with open(output_file, 'w', encoding='utf-8') as game_fh:
                    game_fh.write(current_game + '\n\n')
                local_kept += 1
    
    # Final update
    with lock:
        processed_counter.value += local_processed
    
    return local_kept


def count_total_lines(input_file):
    """Count total lines in a file for chunking."""
    count = 0
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            count += 1
    return count


def filter_games_by_rating_and_time_control_parallel(input_file, output_directory, min_rating=2500, offset=0, num_processes=None):
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
    
    # Count total lines instead of games for better chunking
    print("Counting lines in file...")
    total_lines = count_total_lines(input_file)
    lines_per_process = total_lines // num_processes
    chunks = []
    
    for i in range(num_processes):
        start_line = i * lines_per_process + 1
        if i == num_processes - 1:
            end_line = total_lines
        else:
            end_line = (i + 1) * lines_per_process
        
        chunks.append((input_file, output_directory, start_line, end_line, min_rating, offset, 
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


def filter_games_by_rating_and_time_control(input_file, output_directory, min_rating=2500, offset=0):
    """Filter games where both players have rating >= min_rating using string-based approach."""
    
    games_processed = 0
    games_kept = 0
    
    print(f"Filtering games with both players rated >= {min_rating}...")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fh:
        current_game = ""
        in_game = False
        i = offset
        
        for line in fh:
            line = line.rstrip('\n\r')
            
            if line.startswith('[Event'):
                # Start of new game
                if current_game and in_game:
                    # Process previous game
                    games_processed += 1
                    if extract_and_verify_rating(current_game, min_rating):
                        output_file = os.path.join(output_directory, f'{i}.pgn')
                        with open(output_file, 'w', encoding='utf-8') as game_fh:
                            game_fh.write(current_game + '\n\n')
                        games_kept += 1
                        i += 1
                    
                    if games_kept % 1000 == 0 and games_kept > 0:
                        print(f"Kept {games_kept} games out of {games_processed} games")
                    
                    if games_processed % 10000 == 0 and games_processed > 0:
                        print(f"Processed {games_processed} games out of approximately 3.8MM games")
                    
                    # There are approximately 3.83MM games here
                    if games_processed >= 3830000:
                        break
                
                current_game = line + '\n'
                in_game = True
            elif in_game:
                current_game += line + '\n'
        
        # Process last game
        if current_game and in_game and games_processed < 3830000:
            games_processed += 1
            if extract_and_verify_rating(current_game, min_rating):
                output_file = os.path.join(output_directory, f'{i}.pgn')
                with open(output_file, 'w', encoding='utf-8') as game_fh:
                    game_fh.write(current_game + '\n\n')
                games_kept += 1
    
    print(f"\nFiltering complete!")
    print(f"Total games processed: {games_processed:,}")
    print(f"Games kept (both players >= {min_rating}): {games_kept:,}")
    if games_processed > 0:
        print(f"Percentage kept: {games_kept/games_processed*100:.1f}%")
    
    return games_kept, games_processed


def main():
    parser = argparse.ArgumentParser(description="Download and filter grandmaster-level games from ComputerChess.org.uk")
    parser.add_argument('--offset', type=int, default=0, help='Skip the processing of the first N games')
    parser.add_argument('--min-rating', type=int, default=2500, help='Minimum rating for both players (default: 2500)')
    parser.add_argument('--output-dir', default='games_training_data/reformatted', help='Output directory (default: games_training_data/reformatted)')
    parser.add_argument('--skip-download', action='store_true', help='Skip download and only filter existing files')
    parser.add_argument('--output-dir-downloads', default='games_training_data/CCRL_computerchess_org/', help='Output directory to store LiChess Databases (default: games_training_data)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use for parallel filtering (default: number of CPU cores, max 100)')

    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir_downloads, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we already have enough processed games
    existing_games = count_existing_games_in_directory(args.output_dir)
    print(f"\nFound {existing_games:,} existing games in output directory: {args.output_dir}")
    
    # Estimate expected games: ~3.83M total games, ~65% pass 2500+ filter
    estimated_total_games = 3830000
    expected_filtered_games = int(estimated_total_games * 0.65)
    
    if existing_games >= expected_filtered_games:
        print(f"✓ Already have {existing_games:,} games (expected ~{expected_filtered_games:,} after filtering).")
        print(f"Skipping download and processing.\n")
        print(f"To re-download and reprocess, either:")
        print(f"  1. Delete the existing games in {args.output_dir}")
        print(f"  2. Use --skip-download to process only existing downloaded files")
        return
    elif existing_games > expected_filtered_games * 0.9:  # Within 90% of expected
        print(f"✓ Already have {existing_games:,} games (~{existing_games/expected_filtered_games*100:.1f}% of expected).")
        print(f"This appears to be complete. Skipping download and processing.\n")
        return
    
    if not args.skip_download:
        # Check if we need to download based on existing processed games
        if existing_games > 0:
            print(f"\nNote: Already have {existing_games:,} processed games.")
            print(f"Continuing to download and process more games...\n")
        
        # Get available databases
        print("Downloading CCRL dataset...")
        
        url_path = "https://computerchess.org.uk/ccrl/4040/CCRL-4040.[2145878].pgn.7z"
        fileName = "CCRL-4040.[2145878].pgn.7z"
        url_paths_and_fnames = (url_path, fileName)
        download_file(url_paths_and_fnames)



        archive_path = os.path.join(fileName)
        output_directory = os.path.join(args.output_dir_downloads)

    
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
        # For password-protected archives, uncomment the line below and provide the password
        # with py7zr.SevenZipFile(archive_path, mode='r', password=password) as z:
            z.extractall(path=output_directory)
            print(f"Archive '{archive_path}' successfully extracted to '{output_directory}'")


        url_path = "https://computerchess.org.uk/ccrl/404/CCRL-404.[1692642].pgn.7z"
        fileName = "CCRL-404.[1692642].pgn.7z"
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
    
    MIN_RATING = args.min_rating or 2500

    # Create output directory for filtered games
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we already have processed games
    existing_games = count_existing_games_in_directory(args.output_dir)
    print(f"\nFound {existing_games:,} existing games in output directory: {args.output_dir}")
    
    # Estimate maximum games to process (approximately 3.83MM games total)
    estimated_total_games = 3830000
    
    # Calculate expected games based on typical filter rate (~65% kept for 2500+ rating in CCRL)
    expected_filtered_games = int(estimated_total_games * 0.65)
    
    if existing_games >= expected_filtered_games:
        print(f"✓ Already have {existing_games:,} games (expected ~{expected_filtered_games:,} after filtering). Skipping processing.")
        total_kept = existing_games
        total_processed = estimated_total_games
    else:
        if existing_games > 0:
            print(f"Resuming from game {existing_games:,}")
            total_kept = existing_games
        
        # Process all .pgn files in the output directory
        for filename in sorted(os.listdir(args.output_dir_downloads)):
            if filename.endswith('.pgn'):
                input_path_for_extraction = os.path.join(args.output_dir_downloads, filename)
                input_path_for_parsing = input_path_for_extraction
                output_directory = os.path.join(args.output_dir)
                print(f"\nProcessing {filename}...")
                kept, processed = filter_games_by_rating_and_time_control_parallel(input_path_for_parsing, output_directory, args.min_rating, total_kept, args.processes)
                
                total_kept += kept
                total_processed += processed
    
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