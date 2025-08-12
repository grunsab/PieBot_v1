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

def process_single_game(args):
    """Process a single game file for parallel processing."""
    pgn_file, input_dir, output_base_dir = args
    
    input_path = os.path.join(input_dir, pgn_file)
    avg_rating, white_rating, black_rating = extract_ratings(input_path)
    
    if avg_rating is None:
        return (pgn_file, None, 'no_rating')
    
    category = categorize_game(avg_rating)
    
    if category:
        # Create category directory if it doesn't exist
        category_dir = os.path.join(output_base_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Copy file to appropriate category
        output_path = os.path.join(category_dir, pgn_file)
        shutil.copy2(input_path, output_path)
        
        return (pgn_file, category, avg_rating)
    
    return (pgn_file, None, 'out_of_range')

def organize_games(input_dir, output_base_dir, num_workers=None):
    """Organize games from input directory into ELO-based categories."""
    
    # Create output directories
    for category in ELO_RANGES.keys():
        os.makedirs(os.path.join(output_base_dir, category), exist_ok=True)
    
    # Get list of PGN files
    pgn_files = [f for f in os.listdir(input_dir) if f.endswith('.pgn')]
    
    if not pgn_files:
        print(f"No PGN files found in {input_dir}")
        return
    
    print(f"Found {len(pgn_files)} PGN files to process")
    print(f"Organizing into categories: {list(ELO_RANGES.keys())}")
    print(f"ELO ranges:")
    for category, (min_elo, max_elo) in ELO_RANGES.items():
        print(f"  {category}: {min_elo}-{max_elo}")
    
    # Prepare arguments for parallel processing
    process_args = [(pgn_file, input_dir, output_base_dir) for pgn_file in pgn_files]
    
    # Process files in parallel
    if num_workers is None:
        num_workers = mp.cpu_count() - 1
    
    num_workers = max(1, min(num_workers, mp.cpu_count()))
    
    print(f"\nProcessing with {num_workers} workers...")
    
    category_counts = {cat: 0 for cat in ELO_RANGES.keys()}
    category_counts['no_rating'] = 0
    category_counts['out_of_range'] = 0
    
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(pgn_files), desc="Organizing games") as pbar:
            for result in pool.imap_unordered(process_single_game, process_args):
                pgn_file, category, info = result
                
                if category:
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
    
    print(f"\nTotal organized: {total_organized:,} games")
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
    organize_games(args.input_dir, args.output_dir, args.workers)

if __name__ == "__main__":
    main()