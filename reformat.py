
import chess.pgn
import os
import sys
import argparse
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool, Manager
import time
from functools import partial
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# The default offset to use to determine a unique PGN filename. Important when processing multiple
# large sets of PGNs from disparate sources.
STARTING_OFFSET_FOR_FILENAME = 0

def count_games_in_pgn_fast(input_file):
    """
    Count games by scanning for Result tags, which is much faster than parsing.
    Each game ends with a Result tag.
    """
    count = 0
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('[Result '):
                count += 1
    return count

def count_games_in_pgn(input_file, force_fast=False):
    """
    Count the total number of games in a PGN file.
    Falls back to full parsing if fast method seems incorrect.
    """
    # First try fast counting
    fast_count = count_games_in_pgn_fast(input_file)
    
    # If forced fast mode or count seems reasonable (around 8000), use it
    if force_fast or (7000 <= fast_count <= 9000):
        return fast_count
    
    # Otherwise fall back to accurate counting
    count = 0
    with open(input_file, 'r') as pgn_fh:
        while chess.pgn.skip_game(pgn_fh):
            count += 1
    return count

def process_pgn_chunk(args):
    """
    Process a chunk of games from a PGN file.
    
    Args:
        args: Tuple of (input_file, output_dir, start_idx, end_idx, offset, counter, lock)
    """
    input_file, output_dir, start_idx, end_idx, offset, counter, lock = args
    
    with open(input_file, 'r') as pgn_fh:
        # Skip to the starting position
        for _ in range(start_idx):
            game = chess.pgn.skip_game(pgn_fh)
            if not game:
                return 0
        
        # Process games in this chunk
        games_written = 0
        for i in range(start_idx, end_idx):
            game = chess.pgn.read_game(pgn_fh)
            if not game:
                break
            
            output_file = os.path.join(output_dir, f'{offset + i + STARTING_OFFSET_FOR_FILENAME}.pgn')
            with open(output_file, 'w') as game_fh:
                print(game, file=game_fh, end='\n\n')
            
            games_written += 1
            
            # Update shared counter
            if counter is not None and lock is not None:
                with lock:
                    counter.value += 1
                    if counter.value % 1000 == 0 and not TQDM_AVAILABLE:
                        print(f'Processed {counter.value} games')
    
    return games_written

def reformat_single_pgn(input_file, output_dir, num_processes=None, approx_games=None):
    """
    Reformat a single large PGN file into individual game files using multiprocessing.
    
    Args:
        input_file: Path to the input PGN file
        output_dir: Directory to write individual game files
        num_processes: Number of processes to use (default: CPU count)
        approx_games: Approximate number of games (to skip counting)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Reformatting {input_file} to {output_dir} using {num_processes} processes")
    
    # Count games if not provided
    if approx_games is None:
        print("Counting games in file...")
        start_count = time.time()
        total_games = count_games_in_pgn(input_file)
        count_time = time.time() - start_count
        print(f"Found {total_games} games to process (counting took {count_time:.2f}s)")
    else:
        total_games = approx_games
        print(f"Using approximate game count: {total_games}")
    
    if total_games == 0:
        print("No games found in file")
        return
    
    # For small files, use single process
    if total_games < 1000:
        print("Using single process for small file")
        with open(input_file, 'r') as pgn_fh:
            game_idx = 0
            while True:
                game = chess.pgn.read_game(pgn_fh)
                if not game:
                    break
                
                output_file = os.path.join(output_dir, f'{game_idx + STARTING_OFFSET_FOR_FILENAME}.pgn')
                with open(output_file, 'w') as game_fh:
                    print(game, file=game_fh, end='\n\n')
                
                game_idx += 1
        print(f'Completed reformatting: {game_idx} total games')
        return
    
    # Setup shared counter for progress tracking
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    # Divide work among processes
    games_per_process = total_games // num_processes
    chunks = []
    
    for i in range(num_processes):
        start_idx = i * games_per_process
        if i == num_processes - 1:
            end_idx = total_games
        else:
            end_idx = (i + 1) * games_per_process
        
        chunks.append((input_file, output_dir, start_idx, end_idx, 0, counter, lock))
    
    # Process chunks in parallel
    start_time = time.time()
    
    if TQDM_AVAILABLE:
        with Pool(num_processes) as pool:
            with tqdm(total=total_games, desc="Processing games") as pbar:
                results = []
                for chunk in chunks:
                    result = pool.apply_async(process_pgn_chunk, (chunk,))
                    results.append(result)
                
                # Monitor progress
                last_count = 0
                while any(not r.ready() for r in results):
                    current_count = counter.value
                    pbar.update(current_count - last_count)
                    last_count = current_count
                    time.sleep(0.1)
                
                # Final update
                pbar.update(counter.value - last_count)
                
                # Get results
                results = [r.get() for r in results]
    else:
        with Pool(num_processes) as pool:
            results = pool.map(process_pgn_chunk, chunks)
    
    total_written = sum(results)
    elapsed_time = time.time() - start_time
    
    print(f'Completed reformatting: {total_written} total games in {elapsed_time:.2f} seconds')
    print(f'Processing speed: {total_written/elapsed_time:.2f} games/second')

def process_pgn_file(args):
    """
    Process a single PGN file and return games processed.
    
    Args:
        args: Tuple of (input_path, output_dir, start_offset, counter, lock)
    """
    input_path, output_dir, start_offset, counter, lock = args
    
    games_written = 0
    with open(input_path, 'r') as pgn_fh:
        while True:
            game = chess.pgn.read_game(pgn_fh)
            if not game:
                break
            
            output_file = os.path.join(output_dir, f'{start_offset + games_written + STARTING_OFFSET_FOR_FILENAME}.pgn')
            with open(output_file, 'w') as game_fh:
                print(game, file=game_fh, end='\n\n')
            
            games_written += 1
            
            # Update shared counter
            if counter is not None and lock is not None:
                with lock:
                    counter.value += 1
    
    return games_written

def count_games_parallel(args):
    """
    Count games in a single file for parallel processing.
    """
    pgn_file, input_dir = args
    input_path = os.path.join(input_dir, pgn_file)
    game_count = count_games_in_pgn(input_path)
    return (pgn_file, input_path, game_count)

def reformat_directory(input_dir, output_dir, num_processes=None):
    """
    Reformat all PGN files in a directory using multiprocessing.
    
    Args:
        input_dir: Directory containing PGN files
        output_dir: Directory to write reformatted files
        num_processes: Number of processes to use (default: CPU count)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    pgn_files = [f for f in os.listdir(input_dir) if f.endswith('.pgn')]
    print(f"Found {len(pgn_files)} PGN files in {input_dir}")
    print(f"Using {num_processes} processes")
    
    if not pgn_files:
        print("No PGN files found")
        return
    
    # Count games in each file in parallel
    print("Counting games in files...")
    start_count_time = time.time()
    
    # Prepare arguments for parallel counting
    count_args = [(pgn_file, input_dir) for pgn_file in pgn_files]
    
    # Count games in parallel
    with Pool(num_processes) as pool:
        if TQDM_AVAILABLE:
            count_results = list(tqdm(
                pool.imap(count_games_parallel, count_args),
                total=len(pgn_files),
                desc="Counting games"
            ))
        else:
            count_results = pool.map(count_games_parallel, count_args)
    
    # Process results
    file_info = []
    total_games = 0
    for pgn_file, input_path, game_count in count_results:
        file_info.append((input_path, output_dir, total_games))
        total_games += game_count
        if not TQDM_AVAILABLE:
            print(f"{pgn_file}: {game_count} games")
    
    count_time = time.time() - start_count_time
    print(f"Game counting completed in {count_time:.2f} seconds")
    print(f"Total games to process: {total_games}")
    
    # Setup shared counter for progress tracking
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    # Add counter and lock to file info
    file_info_with_progress = [(path, out_dir, total_games, counter, lock) 
                               for path, out_dir, total_games in file_info]
    
    # Process files in parallel
    start_time = time.time()
    
    if TQDM_AVAILABLE:
        with Pool(num_processes) as pool:
            with tqdm(total=total_games, desc="Processing games") as pbar:
                results = []
                for file_args in file_info_with_progress:
                    result = pool.apply_async(process_pgn_file, (file_args,))
                    results.append(result)
                
                # Monitor progress
                last_count = 0
                while any(not r.ready() for r in results):
                    current_count = counter.value
                    pbar.update(current_count - last_count)
                    last_count = current_count
                    time.sleep(0.1)
                
                # Final update
                pbar.update(counter.value - last_count)
                
                # Get results
                results = [r.get() for r in results]
    else:
        with Pool(num_processes) as pool:
            results = pool.map(process_pgn_file, file_info_with_progress)
    
    total_written = sum(results)
    elapsed_time = time.time() - start_time
    
    print(f'Completed reformatting: {total_written} total games in {elapsed_time:.2f} seconds')
    print(f'Processing speed: {total_written/elapsed_time:.2f} games/second')

def main():
    parser = argparse.ArgumentParser(description='Reformat PGN files for AlphaZero training')
    parser.add_argument('input', help='Input PGN file or directory')
    parser.add_argument('output', help='Output directory for reformatted files', default='games_training_data/reformatted/')
    parser.add_argument('--processes', '-p', type=int, default=None,
                        help='Number of processes to use (default: CPU count)')
    parser.add_argument('--approx-games', '-a', type=int, default=8000,
                        help='Approximate number of games per file (default: 8000, use 0 to force counting)')
    parser.add_argument('--fast-count', '-f', action='store_true',
                        help='Use fast game counting by scanning Result tags')
    parser.add_argument("--starting-offset", type=int, default=0, help="Set starting index for PGN file number")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)
    
    if args.starting_offset:
        STARTING_OFFSET_FOR_FILENAME = args.starting_offset


    if input_path.is_file():
        approx = args.approx_games if args.approx_games > 0 else None
        reformat_single_pgn(str(input_path), str(output_path), args.processes, approx)
    elif input_path.is_dir():
        reformat_directory(str(input_path), str(output_path), args.processes)
    else:
        print(f"Error: {input_path} is neither a file nor directory")
        sys.exit(1)

if __name__ == "__main__":
    main()

