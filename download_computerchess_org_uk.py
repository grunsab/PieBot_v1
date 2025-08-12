#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and extract chess games from ComputerChess.org.uk.
Optimized for fast extraction without filtering.
"""

import requests
import os
import sys
import time
import argparse
from tqdm import tqdm
import py7zr

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


def extract_games_ultra_fast(input_file, output_directory, offset=0):
    """Extract all games with batch writing for maximum speed."""
    
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Extracting games from {os.path.basename(input_file)}...")
    
    games_processed = 0
    start_time = time.time()
    last_report_time = start_time
    
    # Batch parameters
    batch_size = 10000  # Write 10k games at once
    games_batch = []
    game_idx = offset
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as fh:
        current_game = []
        in_game = False
        
        for line in fh:
            if line.startswith('[Event'):
                # Start of new game
                if current_game and in_game:
                    # Add to batch
                    games_batch.append((''.join(current_game) + '\n', game_idx))
                    games_processed += 1
                    game_idx += 1
                    
                    # Write batch when full
                    if len(games_batch) >= batch_size:
                        for game_content, idx in games_batch:
                            output_file = os.path.join(output_directory, f'{idx}.pgn')
                            with open(output_file, 'w', encoding='utf-8') as game_fh:
                                game_fh.write(game_content)
                        
                        # Report progress
                        current_time = time.time()
                        elapsed = current_time - start_time
                        speed = games_processed / elapsed if elapsed > 0 else 0
                        print(f"  Processed: {games_processed:,} games, Speed: {speed:.0f} games/sec")
                        
                        games_batch = []
                        last_report_time = current_time
                    
                    current_game = []
                
                current_game.append(line)
                in_game = True
            elif in_game:
                current_game.append(line)
        
        # Process last game
        if current_game and in_game:
            games_batch.append((''.join(current_game) + '\n', game_idx))
            games_processed += 1
        
        # Write remaining batch
        if games_batch:
            for game_content, idx in games_batch:
                output_file = os.path.join(output_directory, f'{idx}.pgn')
                with open(output_file, 'w', encoding='utf-8') as game_fh:
                    game_fh.write(game_content)
    
    elapsed_time = time.time() - start_time
    
    print(f"  Extraction complete!")
    print(f"  Total games extracted: {games_processed:,}")
    print(f"  Processing time: {elapsed_time:.2f} seconds")
    print(f"  Processing speed: {games_processed/elapsed_time:.0f} games/second")
    
    return games_processed


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


def main():
    parser = argparse.ArgumentParser(description="Download and extract games from ComputerChess.org.uk")
    parser.add_argument('--output-dir', default='games_training_data/reformatted', 
                       help='Output directory (default: games_training_data/reformatted)')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip download and only process existing files')
    parser.add_argument('--output-dir-downloads', default='games_training_data/CCRL_computerchess_org/', 
                       help='Output directory to store downloaded PGN files (default: games_training_data/CCRL_computerchess_org/)')

    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir_downloads, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we already have enough processed games
    existing_games = count_existing_games_in_directory(args.output_dir)
    print(f"\nFound {existing_games:,} existing games in output directory: {args.output_dir}")
    
    # Estimate expected games: ~3.83M total games
    estimated_total_games = 3830000
    
    if existing_games >= estimated_total_games:
        print(f"✓ Already have {existing_games:,} games (expected ~{estimated_total_games:,} total).")
        print(f"Skipping download and processing.\n")
        print(f"To re-download and reprocess, either:")
        print(f"  1. Delete the existing games in {args.output_dir}")
        print(f"  2. Use --skip-download to process only existing downloaded files")
        return
    elif existing_games > estimated_total_games * 0.9:  # Within 90% of expected
        print(f"✓ Already have {existing_games:,} games (~{existing_games/estimated_total_games*100:.1f}% of expected).")
        print(f"This appears to be complete. Skipping download and processing.\n")
        return
    
    if not args.skip_download:
        # Check if we need to download based on existing processed games
        if existing_games > 0:
            print(f"\nNote: Already have {existing_games:,} processed games.")
            print(f"Continuing to download and process more games...\n")
        
        # Download CCRL datasets
        print("Downloading CCRL dataset...")
        
        # Download first dataset
        url_path = "https://computerchess.org.uk/ccrl/4040/CCRL-4040.[2145878].pgn.7z"
        fileName = "CCRL-4040.[2145878].pgn.7z"
        url_paths_and_fnames = (url_path, fileName)
        download_file(url_paths_and_fnames)

        archive_path = os.path.join(fileName)
        output_directory = os.path.join(args.output_dir_downloads)
    
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            z.extractall(path=output_directory)
            print(f"Archive '{archive_path}' successfully extracted to '{output_directory}'")

        # Download second dataset
        url_path = "https://computerchess.org.uk/ccrl/404/CCRL-404.[1692642].pgn.7z"
        fileName = "CCRL-404.[1692642].pgn.7z"
        url_paths_and_fnames = (url_path, fileName)
        download_file(url_paths_and_fnames)

        archive_path = os.path.join(fileName)
        output_directory = os.path.join(args.output_dir_downloads)

        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            z.extractall(path=output_directory)
            print(f"Archive '{archive_path}' successfully extracted to '{output_directory}'")
    
    # Process extracted files
    total_extracted = 0
    
    # Check if we already have processed games
    existing_games = count_existing_games_in_directory(args.output_dir)
    print(f"\nFound {existing_games:,} existing games in output directory: {args.output_dir}")
    
    if existing_games >= estimated_total_games:
        print(f"✓ Already have {existing_games:,} games (expected ~{estimated_total_games:,} total). Skipping processing.")
        total_extracted = existing_games
    else:
        if existing_games > 0:
            print(f"Resuming from game {existing_games:,}")
            total_extracted = existing_games
        
        # Process all .pgn files in the downloads directory
        for filename in sorted(os.listdir(args.output_dir_downloads)):
            if filename.endswith('.pgn'):
                input_path = os.path.join(args.output_dir_downloads, filename)
                print(f"\nProcessing {filename}...")
                extracted = extract_games_ultra_fast(input_path, args.output_dir, total_extracted)
                total_extracted += extracted
    
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Total games extracted: {total_extracted:,}")
    print(f"Output directory: {args.output_dir}/")

if __name__ == "__main__":
    main()