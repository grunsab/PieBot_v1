#!/usr/bin/env python3
"""
Test script to verify Lichess download and filtering works correctly.
"""

import os
import sys

# Test the download and filter with minimal data
print("Testing Lichess download and filtering...")
print("="*50)

# Run the download script with test parameters
# Skip download and only test filtering if a file already exists
test_cmd = """
python3 download_lichess_games.py \
    --months 1 \
    --min-rating 750 \
    --output-dir games_training_data/test_lichess \
    --output-dir-downloads games_training_data/LiChessData/ \
    --skip-download
"""

print("Running command:")
print(test_cmd)

result = os.system(test_cmd)

if result == 0:
    # Check if any files were created
    output_dir = "games_training_data/test_lichess"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        pgn_files = [f for f in files if f.endswith('.pgn')]
        print(f"\nTest successful! Found {len(pgn_files)} filtered game files.")
        if pgn_files:
            print(f"Sample files: {pgn_files[:5]}")
    else:
        print("\nWarning: Output directory not created.")
else:
    print(f"\nError: Command failed with code {result}")

print("="*50)
print("Test complete.")