#!/usr/bin/env python3
"""
Organize chess games by ELO rating ranges for curriculum training.
This script takes PGN files and sorts them into directories based on average player ratings.
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Define ELO ranges for curriculum stages
ELO_RANGES = {
    'beginner': (750, 1500),      # Stage 1: Basic piece values and tactics
    'intermediate': (1500, 2400),  # Stage 2: Positional understanding
    'expert': (2400, 3000),        # Stage 3: Strong human and titled players
    'computer': (3000, 4000)       # Stage 4: Computer chess engines
}

def extract_ratings(pgn_file_path):
    """Extract white and black ratings from a PGN file."""
    try:
        with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        white_rating = None
        black_rating = None
        
        # Search for ratings in the PGN headers
        white_match = re.search(r'\[WhiteElo "(\d+)"\]', content)
        black_match = re.search(r'\[BlackElo "(\d+)"\]', content)
        
        if white_match:
            white_rating = int(white_match.group(1))
        if black_match:
            black_rating = int(black_match.group(1))
            
        if white_rating and black_rating:
            avg_rating = (white_rating + black_rating) // 2
            return avg_rating, white_rating, black_rating
    except Exception as e:
        print(f"Error processing {pgn_file_path}: {e}")
    
    return None, None, None

def categorize_game(avg_rating):
    """Determine which category a game belongs to based on average rating."""
    if avg_rating is None:
        return None
        
    for category, (min_elo, max_elo) in ELO_RANGES.items():
        if min_elo <= avg_rating < max_elo:
            return category
    
    # If above all ranges, put in expert category
    if avg_rating >= ELO_RANGES['expert'][0]:
        return 'expert'
    
    return None


def count_single_game_category(args):
    """Count category for a single game file (for parallel processing)."""
    pgn_file, input_dir = args
    input_path = os.path.join(input_dir, pgn_file)
    avg_rating, _, _ = extract_ratings(input_path)
    
    if avg_rating is None:
        return 'no_rating'
    else:
        category = categorize_game(avg_rating)
        if category:
            return category
        else:
            return 'out_of_range'


def quick_count_by_category(input_dir, num_workers=None):
    """Quickly count how many games should be in each category without copying files."""
    pgn_files = [f for f in os.listdir(input_dir) if f.endswith('.pgn')]
    
    if not pgn_files:
        return None
    
    print(f"Quick counting {len(pgn_files):,} PGN files by ELO category...")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count() - 1, 32)  # Cap at 32 for counting
    num_workers = max(1, num_workers)
    
    print(f"Using {num_workers} workers for parallel counting...")
    
    category_counts = {cat: 0 for cat in ELO_RANGES.keys()}
    category_counts['no_rating'] = 0
    category_counts['out_of_range'] = 0
    
    # Prepare arguments for parallel processing
    process_args = [(pgn_file, input_dir) for pgn_file in pgn_files]
    
    # Process files in parallel
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(pgn_files), desc="Counting games") as pbar:
            for category in pool.imap_unordered(count_single_game_category, process_args, chunksize=100):
                category_counts[category] += 1
                pbar.update(1)
    
    return category_counts


def count_existing_organized_games(output_base_dir):
    """Count games already organized in each category directory."""
    existing_counts = {cat: 0 for cat in ELO_RANGES.keys()}
    existing_files = {cat: set() for cat in ELO_RANGES.keys()}
    
    for category in ELO_RANGES.keys():
        category_dir = os.path.join(output_base_dir, category)
        if os.path.exists(category_dir):
            pgn_files = [f for f in os.listdir(category_dir) if f.endswith('.pgn')]
            existing_counts[category] = len(pgn_files)
            existing_files[category] = set(pgn_files)
    
    return existing_counts, existing_files


def get_input_source_identifier(input_dir):
    """Get a unique identifier for the input source based on directory name."""
    if 'lichess' in input_dir.lower():
        return 'lichess'
    elif 'reformatted' in input_dir and 'lichess' not in input_dir.lower():
        return 'ccrl'
    else:
        # Use last directory name as identifier
        return os.path.basename(os.path.normpath(input_dir))

def check_if_already_organized(input_dir, output_base_dir, num_workers=None):
    """Check if games from this specific input source have already been organized."""
    source_id = get_input_source_identifier(input_dir)
    print(f"\nChecking if games from '{source_id}' source are already organized...")
    
    # Get list of input files
    input_files = set(f for f in os.listdir(input_dir) if f.endswith('.pgn'))
    
    if not input_files:
        print(f"No PGN files found in {input_dir}")
        return False, None, None
    
    # Quick count what should be in each category (with parallel processing)
    print(f"Analyzing {len(input_files):,} files from {input_dir}...")
    expected_counts = quick_count_by_category(input_dir, num_workers)
    
    if expected_counts is None:
        return False, None, None
    
    # Count what's actually in each category
    existing_counts, existing_files = count_existing_organized_games(output_base_dir)
    
    # Check how many of THIS source's files are already organized
    already_organized_from_source = {cat: 0 for cat in ELO_RANGES.keys()}
    
    for category in ELO_RANGES.keys():
        # Count how many files from this input source are already in the output
        for input_file in input_files:
            if input_file in existing_files[category]:
                already_organized_from_source[category] += 1
    
    # Compare counts for this specific source
    all_match = True
    for category in ELO_RANGES.keys():
        expected = expected_counts[category]
        already_organized = already_organized_from_source[category]
        if expected != already_organized:
            all_match = False
            break
    
    # Display comparison
    print("\n" + "="*75)
    print(f"Source: {source_id}")
    print("="*75)
    print("Category".ljust(15) + "Expected".rjust(12) + "From Source".rjust(12) + "Total Existing".rjust(15) + "Status".rjust(15))
    print("-"*75)
    
    for category in ELO_RANGES.keys():
        expected = expected_counts[category]
        from_source = already_organized_from_source[category]
        total_existing = existing_counts[category]
        status = "✓ Done" if expected == from_source else f"Need: {expected - from_source}"
        min_elo, max_elo = ELO_RANGES[category]
        cat_label = f"{category[:12]}"
        print(f"{cat_label.ljust(15)}{expected:12,}{from_source:12,}{total_existing:15,}  {status}")
    
    print("="*75)
    
    total_expected = sum(expected_counts[cat] for cat in ELO_RANGES.keys())
    total_from_source = sum(already_organized_from_source[cat] for cat in ELO_RANGES.keys())
    total_existing = sum(existing_counts[cat] for cat in ELO_RANGES.keys())
    
    print(f"Total".ljust(15) + f"{total_expected:12,}" + f"{total_from_source:12,}" + f"{total_existing:15,}")
    
    if all_match:
        print(f"\n✓ All games from '{source_id}' are already organized. Skipping.")
        return True, expected_counts, existing_counts
    elif total_from_source > 0:
        print(f"\n⚠ Partial: {total_from_source:,}/{total_expected:,} games from '{source_id}' already organized")
        print(f"Will organize remaining {total_expected - total_from_source:,} games...")
        return False, expected_counts, existing_counts
    else:
        print(f"\n✗ No games from '{source_id}' have been organized yet.")
        return False, expected_counts, existing_counts

def process_single_game(args):
    """Process a single game file for parallel processing."""
    pgn_file, input_dir, output_base_dir, skip_existing = args
    
    input_path = os.path.join(input_dir, pgn_file)
    avg_rating, white_rating, black_rating = extract_ratings(input_path)
    
    if avg_rating is None:
        return (pgn_file, None, 'no_rating', False)
    
    category = categorize_game(avg_rating)
    
    if category:
        # Create category directory if it doesn't exist
        category_dir = os.path.join(output_base_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Check if file already exists
        output_path = os.path.join(category_dir, pgn_file)
        
        if skip_existing and os.path.exists(output_path):
            # File already organized, skip copying
            return (pgn_file, category, avg_rating, True)
        
        # Copy file to appropriate category
        shutil.copy2(input_path, output_path)
        
        return (pgn_file, category, avg_rating, False)
    
    return (pgn_file, None, 'out_of_range', False)

def organize_games(input_dir, output_base_dir, num_workers=None, force=False):
    """Organize games from input directory into ELO-based categories."""
    
    # Check if already organized
    if not force:
        already_organized, expected_counts, existing_counts_dict = check_if_already_organized(input_dir, output_base_dir, num_workers)
        
        if already_organized:
            print("\nSkipping organization - games from this source are already organized.")
            # Return existing counts (first element of tuple)
            if isinstance(existing_counts_dict, tuple):
                return existing_counts_dict[0]
            return existing_counts_dict
    
    # Create output directories
    for category in ELO_RANGES.keys():
        os.makedirs(os.path.join(output_base_dir, category), exist_ok=True)
    
    # Get list of PGN files
    pgn_files = [f for f in os.listdir(input_dir) if f.endswith('.pgn')]
    
    if not pgn_files:
        print(f"No PGN files found in {input_dir}")
        return
    
    print(f"\nFound {len(pgn_files)} PGN files to process")
    print(f"Organizing into categories: {list(ELO_RANGES.keys())}")
    print(f"ELO ranges:")
    for category, (min_elo, max_elo) in ELO_RANGES.items():
        print(f"  {category}: {min_elo}-{max_elo}")
    
    # Check for existing files to avoid overwriting
    skip_existing = True  # Always skip existing files when merging from multiple sources
    
    # Prepare arguments for parallel processing
    process_args = [(pgn_file, input_dir, output_base_dir, skip_existing) for pgn_file in pgn_files]
    
    # Process files in parallel
    if num_workers is None:
        num_workers = mp.cpu_count() - 1
    
    num_workers = max(1, min(num_workers, mp.cpu_count()))
    
    print(f"\nProcessing with {num_workers} workers...")
    if skip_existing:
        print("(Skipping files that are already organized)")
    
    category_counts = {cat: 0 for cat in ELO_RANGES.keys()}
    category_counts['no_rating'] = 0
    category_counts['out_of_range'] = 0
    category_counts['skipped'] = 0
    
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(pgn_files), desc="Organizing games") as pbar:
            for result in pool.imap_unordered(process_single_game, process_args):
                pgn_file, category, info, was_skipped = result
                
                if was_skipped:
                    category_counts['skipped'] += 1
                elif category:
                    category_counts[category] += 1
                elif info == 'no_rating':
                    category_counts['no_rating'] += 1
                elif info == 'out_of_range':
                    category_counts['out_of_range'] += 1
                
                pbar.update(1)
    
    # Print summary
    print("\n" + "="*50)
    print("ORGANIZATION SUMMARY")
    print("="*50)
    
    total_organized = sum(category_counts[cat] for cat in ELO_RANGES.keys())
    
    for category in ELO_RANGES.keys():
        count = category_counts[category]
        min_elo, max_elo = ELO_RANGES[category]
        print(f"{category.capitalize()} ({min_elo}-{max_elo} ELO): {count:,} games")
    
    print(f"\nTotal newly organized: {total_organized:,} games")
    if category_counts['skipped'] > 0:
        print(f"Already organized (skipped): {category_counts['skipped']:,} games")
    print(f"No rating found: {category_counts['no_rating']:,} games")
    print(f"Out of range: {category_counts['out_of_range']:,} games")
    
    # Create summary file
    summary_path = os.path.join(output_base_dir, 'organization_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Game Organization Summary\n")
        f.write("="*50 + "\n\n")
        
        for category in ELO_RANGES.keys():
            count = category_counts[category]
            min_elo, max_elo = ELO_RANGES[category]
            f.write(f"{category.capitalize()} ({min_elo}-{max_elo} ELO): {count:,} games\n")
        
        f.write(f"\nTotal organized: {total_organized:,} games\n")
        f.write(f"No rating found: {category_counts['no_rating']:,} games\n")
        f.write(f"Out of range: {category_counts['out_of_range']:,} games\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    return category_counts

def main():
    parser = argparse.ArgumentParser(
        description="Organize chess games by ELO rating for curriculum training"
    )
    parser.add_argument(
        '--input-dir', 
        default='games_training_data/reformatted_lichess',
        help='Directory containing PGN files to organize'
    )
    parser.add_argument(
        '--output-dir', 
        default='games_training_data/curriculum',
        help='Base directory for organized output'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )
    parser.add_argument(
        '--custom-ranges',
        action='store_true',
        help='Use custom ELO ranges (will prompt for input)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-organization even if games are already organized'
    )
    
    args = parser.parse_args()
    
    # Allow custom ELO ranges if requested
    if args.custom_ranges:
        print("Enter custom ELO ranges (leave blank to use defaults):")
        
        for category in ['beginner', 'intermediate', 'expert']:
            current_min, current_max = ELO_RANGES[category]
            
            try:
                min_input = input(f"{category} min ELO [{current_min}]: ").strip()
                max_input = input(f"{category} max ELO [{current_max}]: ").strip()
                
                if min_input:
                    ELO_RANGES[category] = (int(min_input), ELO_RANGES[category][1])
                if max_input:
                    ELO_RANGES[category] = (ELO_RANGES[category][0], int(max_input))
            except ValueError:
                print(f"Invalid input, using defaults for {category}")
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Organize the games
    organize_games(args.input_dir, args.output_dir, args.workers, args.force)

if __name__ == "__main__":
    main()