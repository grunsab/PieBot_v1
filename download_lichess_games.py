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
MAX_GAMES_TO_COLLECT = 90000000  # 90M games maximum


import uuid
import zstandard as zstd
from collections import defaultdict



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


def count_existing_games_in_directory(directory, prefix='lichess_'):
    """
    Count the number of already processed games in the output directory.
    Games are expected to be named with a specific prefix and number pattern.
    """
    if not os.path.exists(directory):
        return 0
    
    count = 0
    pattern = f'{prefix}*.pgn'
    
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.pgn'):
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

def validate_file(filepath, expected_size_gb=30, tolerance_gb=5):
    """
    Validate that a downloaded file exists and is approximately the expected size.
    
    Args:
        filepath: Path to the file to validate
        expected_size_gb: Expected file size in GB (default: 30)
        tolerance_gb: Tolerance for size validation in GB (default: 5)
    
    Returns:
        bool: True if file exists and is within expected size range
    """
    if not os.path.exists(filepath):
        return False
    
    file_size_bytes = os.path.getsize(filepath)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
    
    min_size = expected_size_gb - tolerance_gb
    max_size = expected_size_gb + tolerance_gb
    
    if file_size_gb < min_size or file_size_gb > max_size:
        print(f"  Warning: {os.path.basename(filepath)} size is {file_size_gb:.1f}GB, expected ~{expected_size_gb}GB")
        return False
    
    # Try to open the file with zstandard to verify it's not corrupt
    try:
        with open(filepath, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                # Read just a small chunk to verify the file is valid
                reader.read(1024)
        return True
    except Exception as e:
        print(f"  Warning: {os.path.basename(filepath)} appears to be corrupt: {e}")
        return False

def download_file(url_filename_pairs, chunk_size=8192):
    """Download a file with progress bar."""
    url = url_filename_pairs[0]
    filepath = url_filename_pairs[1]
    
    # Check if file already exists and is valid
    if validate_file(filepath):
        print(f"  ✓ {os.path.basename(filepath)} already exists and is valid (skipping download)")
        return
    
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


def extract_and_verify_rating(pgn_headers, min_rating=MIN_RATING):
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

def extract_and_verify_time_controls(pgn_headers, min_seconds=180):
    """Extract and verify time control from PGN headers."""

    time_control = None

    time_control_match = re.search(r'\[TimeControl "(\d+)\+(\d+)"\]', pgn_headers)

    if time_control_match:
        initial_time, increment = time_control_match.groups()
    else:
        return False

    return int(initial_time) >= min_seconds




def process_game_batch(args):
    """
    Process a batch of complete games in parallel.
    Optimized to use pre-allocated indices and batch writes.
    
    Args:
        args: Tuple of (games_batch, output_dir, min_rating, min_seconds, base_idx)
    """
    games_batch, output_dir, min_rating, min_seconds, base_idx = args
    
    local_kept = 0
    local_processed = 0
    current_idx = base_idx
    
    # Process all games and collect those to keep
    games_to_write = []
    
    for i, game_text in enumerate(games_batch):
        local_processed += 1
        rating_check = extract_and_verify_rating(game_text, min_rating)
        time_control_check = extract_and_verify_time_controls(game_text, min_seconds)
        
        if rating_check and time_control_check:
            # Use pre-allocated index based on position in batch
            # This ensures no conflicts between parallel workers
            game_idx = base_idx + i
            game_filename = f"lichess_{game_idx:06d}.pgn"
            game_filepath = os.path.join(output_dir, game_filename)
            games_to_write.append((game_filepath, game_text))
            local_kept += 1
    
    # Write all games in this batch
    for filepath, content in games_to_write:
        with open(filepath, 'w', encoding='utf-8') as game_file:
            game_file.write(content + '\n\n')
    
    return local_kept, local_processed


def filter_games_by_rating_and_time_control_parallel(input_file, output_dir, min_rating=750, min_seconds=180, 
                                                    max_games=MAX_GAMES_TO_COLLECT, num_processes=None):
    """
    Filter games using single decompression stream with parallel game processing.
    Optimized version that minimizes file I/O contention and directory scanning.
    """
    
    if num_processes is None:
        # Optimal for I/O-bound tasks is much lower than CPU count
        # 4-8 processes is typically the sweet spot for file I/O operations
        num_processes = min(cpu_count(), 8)  # Cap at 8 for I/O operations
    
    print(f"Filtering games with both players rated >= {min_rating}...")
    print(f"Using optimized parallel processing with {num_processes} workers")
    print(f"Saving individual games to: {output_dir}")
    
    # For single process, use original function
    if num_processes == 1:
        return filter_games_by_rating_and_time_control(input_file, output_dir, min_rating, min_seconds, max_games)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Count existing games ONCE at the start to avoid repeated directory scanning
    existing_games_count = count_existing_games_in_directory(output_dir, 'lichess_')
    print(f"Found {existing_games_count:,} existing games in output directory")
    
    games_processed = 0
    games_kept = 0
    batch_size = 2000  # Increased batch size to reduce synchronization overhead
    
    start_time = time.time()
    
    # Single decompression stream
    with open(input_file, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_reader = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')
            
            current_game = ""
            in_game = False
            games_batch = []
            
            # Create process pool
            with Pool(num_processes) as pool:
                for line in text_reader:
                    line = line.rstrip('\n\r')
                    
                    if line.startswith('[Event'):
                        # Start of new game
                        if current_game and in_game:
                            games_batch.append(current_game)
                            
                            # Process batch when full
                            if len(games_batch) >= batch_size:
                                # Split batch among workers
                                chunk_size = len(games_batch) // num_processes + 1
                                chunks = []
                                
                                # Pre-calculate indices for each chunk to avoid conflicts
                                current_base_idx = existing_games_count + games_kept
                                
                                for i in range(0, len(games_batch), chunk_size):
                                    chunk = games_batch[i:i+chunk_size]
                                    if chunk:
                                        # Each chunk gets its own starting index
                                        chunks.append((chunk, output_dir, min_rating, min_seconds, current_base_idx))
                                        # Reserve indices for this chunk (even if not all games are kept)
                                        current_base_idx += len(chunk)
                                
                                # Process chunks in parallel
                                results = pool.map(process_game_batch, chunks)
                                
                                # Update counters
                                for kept, processed in results:
                                    games_kept += kept
                                    games_processed += processed
                                
                                if games_processed % 10000 == 0:
                                    elapsed = time.time() - start_time
                                    speed = games_processed / elapsed if elapsed > 0 else 0
                                    print(f"Processed: {games_processed:,} games, Kept: {games_kept:,} games "
                                          f"({games_kept/games_processed*100:.1f}%), Speed: {speed:.0f} games/sec")
                                
                                games_batch = []
                                
                                # Check if we've collected enough games
                                if games_kept >= max_games:
                                    break
                        
                        current_game = line + '\n'
                        in_game = True
                    elif in_game:
                        current_game += line + '\n'
                
                # Process remaining games in batch
                if current_game and in_game:
                    games_batch.append(current_game)
                
                if games_batch and games_kept < max_games:
                    # Process final batch
                    chunk_size = len(games_batch) // num_processes + 1
                    chunks = []
                    
                    # Pre-calculate indices for final batch
                    current_base_idx = existing_games_count + games_kept
                    
                    for i in range(0, len(games_batch), chunk_size):
                        chunk = games_batch[i:i+chunk_size]
                        if chunk:
                            # Each chunk gets its own starting index
                            chunks.append((chunk, output_dir, min_rating, min_seconds, current_base_idx))
                            # Reserve indices for this chunk
                            current_base_idx += len(chunk)
                    
                    results = pool.map(process_game_batch, chunks)
                    
                    for kept, processed in results:
                        games_kept += kept
                        games_processed += processed
    
    elapsed_time = time.time() - start_time
    
    print(f"\nFiltering complete!")
    print(f"Total games processed: {games_processed:,}")
    print(f"Games kept (both players >= {min_rating}, time >= {min_seconds}s): {games_kept:,}")
    if games_processed > 0:
        print(f"Percentage kept: {games_kept/games_processed*100:.1f}%")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    if elapsed_time > 0:
        print(f"Processing speed: {games_processed/elapsed_time:.2f} games/second")
    
    return games_kept, games_processed


def filter_games_by_rating_and_time_control(input_file, output_dir, min_rating=750, min_seconds=180, max_games=MAX_GAMES_TO_COLLECT):
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
                        if games_kept >= max_games:
                            break
                    
                    current_game = line + '\n'
                    in_game = True
                elif in_game:
                    current_game += line + '\n'
            
            # Process last game
            if current_game and in_game:
                rating_check = extract_and_verify_rating(current_game, min_rating)
                time_control_check = extract_and_verify_time_controls(current_game, min_seconds)
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
    parser.add_argument('--delete-after-processing', action='store_true', help='Delete compressed files after processing to save space')
    parser.add_argument('--max-games', type=int, default=MAX_GAMES_TO_COLLECT, help=f'Maximum games to collect (default: {MAX_GAMES_TO_COLLECT})')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use for parallel filtering (default: min(CPU cores, 4))')

    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir_downloads, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we already have enough processed games
    existing_games = count_existing_games_in_directory(args.output_dir, 'lichess_')
    print(f"\nFound {existing_games:,} existing games in output directory: {args.output_dir}")
    
    if existing_games >= args.max_games:
        print(f"✓ Already have {existing_games:,} games (max: {args.max_games:,}). Skipping download and processing.")
        print(f"\nTo re-download and reprocess, either:")
        print(f"  1. Delete the existing games in {args.output_dir}")
        print(f"  2. Increase --max-games beyond {args.max_games:,}")
        return
    
    if not args.skip_download:
        # Check if we need to download based on existing processed games
        if existing_games > 0:
            print(f"\nNote: Already have {existing_games:,} processed games.")
            print(f"Will check if download is needed to reach {args.max_games:,} games...")
        
        # Get available databases
        print("\nFetching available Lichess databases...")
        databases = get_available_databases()
        
        if not databases:
            print("No databases found!")
            return
        
        # Download requested number of months
        databases_to_download = databases[:args.months]
        
        print(f"\nFound {len(databases)} databases. Will download {len(databases_to_download)}:")
        for db in databases_to_download:
            print(f"  - {db}")
        
        # Check which files need to be downloaded
        url_paths_and_fnames = []
        all_files_valid = True

        for db_path in databases_to_download:
            filename = os.path.basename(db_path)
            filepath = os.path.join(args.output_dir_downloads, filename)
            
            if validate_file(filepath):
                print(f"  ✓ {filename} already exists and is valid (will skip download)")
            else:
                url = LICHESS_DB_URL + db_path
                url_paths_and_fnames.append((url, filepath))
                all_files_valid = False

        if url_paths_and_fnames:
            print(f"\nDownloading {len(url_paths_and_fnames)} file(s)...")
            parallel_download_mp(url_paths_and_fnames)
        elif all_files_valid:
            print("\n✓ All requested files already exist and are valid. Proceeding to filtering...")
    
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
    # Check if we already have enough processed games
    existing_games = count_existing_games_in_directory(args.output_dir, 'lichess_')
    print(f"\nFound {existing_games:,} existing games in output directory: {args.output_dir}")
    
    if existing_games >= args.max_games:
        print(f"✓ Already have {existing_games:,} games (max: {args.max_games:,}). Skipping processing.")
        total_kept = existing_games
        total_processed = existing_games  # Approximate
    else:
        if existing_games > 0:
            print(f"Need {args.max_games - existing_games:,} more games to reach maximum of {args.max_games:,}")
            total_kept = existing_games
        
        for filename in sorted(zst_files):
            input_path_for_extraction = os.path.join(args.output_dir_downloads, filename)
            input_path_for_parsing = input_path_for_extraction
            # Use output_dir directly as the directory for individual game files
            output_directory = args.output_dir
            print(f"\nProcessing {filename}...")
            print(f"  Input: {input_path_for_parsing}")
            print(f"  Output directory: {output_directory}")
            
            # Calculate how many more games we need
            remaining_games_needed = args.max_games - total_kept
            if remaining_games_needed <= 0:
                print(f"  Already have enough games. Skipping this file.")
                break
            
            # Use parallel processing for filtering
            kept, processed = filter_games_by_rating_and_time_control_parallel(
                input_path_for_parsing, output_directory, 
                min_rating=MIN_RATING, min_seconds=180, max_games=remaining_games_needed,
                num_processes=args.processes
            )
            
            total_kept += kept
            total_processed += processed
            fName_count += 1
            
            # Delete compressed file after processing if requested
            if args.delete_after_processing and kept > 0:
                print(f"  Deleting compressed file to save space: {filename}")
                os.remove(input_path_for_extraction)
            
            # Stop if we've collected enough games globally
            if total_kept >= args.max_games:
                print(f"\nReached maximum games limit ({args.max_games}). Stopping.")
                break
    
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Total games processed: {total_processed:,}")
    print(f"Total games kept: {total_kept:,}")
    if total_processed > 0:
        print(f"Overall percentage: {total_kept/total_processed*100:.1f}%")
    
    # Storage estimate
    if total_kept > 0:
        avg_game_size_kb = 2.5  # Average PGN game size in KB
        storage_gb = (total_kept * avg_game_size_kb) / (1024 * 1024)
        print(f"\nEstimated storage for filtered games: {storage_gb:.1f} GB")
        print(f"(Based on ~{avg_game_size_kb:.1f}KB per game)")
        
        if args.delete_after_processing:
            print("Compressed files deleted to save space.")
        else:
            # Estimate compressed file size based on number of files processed
            compressed_gb = fName_count * 30  # Rough estimate of 30GB per monthly file
            print(f"Compressed files retained: ~{compressed_gb} GB")
            print(f"Total storage used: ~{storage_gb + compressed_gb:.1f} GB")
    
    print(f"\nFiltered games saved in: {args.output_dir}/")

if __name__ == "__main__":
    main()