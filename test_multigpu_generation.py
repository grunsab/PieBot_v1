#!/usr/bin/env python3
"""
Test script for multi-GPU game generation.
Generates a small number of games to verify the system works correctly.
"""

import os
import sys
import subprocess
import time

def test_multigpu_generation():
    """Test multi-GPU game generation with a small number of games"""
    
    print("=" * 60)
    print("Testing Multi-GPU Game Generation")
    print("=" * 60)
    print()
    
    # Check for model
    model_path = "weights/AlphaZeroNet_20x256.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please download or train a model first.")
        return False
    
    # Test parameters
    test_games = 30  # 10 games per GPU
    output_dir = "test_multigpu_output"
    
    # Clean up previous test
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"Test configuration:")
    print(f"  Model: {model_path}")
    print(f"  Total games: {test_games}")
    print(f"  Output directory: {output_dir}")
    print(f"  Format: HDF5 (.h5)")
    print()
    
    # Build command
    cmd = [
        sys.executable,
        "create_training_games_multigpu.py",
        "--model", model_path,
        "--games-total", str(test_games),
        "--rollouts", "10",  # Small for testing
        "--threads-per-gpu", "4",  # Small for testing
        "--save-format", "h5",
        "--output-dir", output_dir,
        "--file-base", "test_game",
        "--temperature", "1.0",
        "--iteration", "1",
        "--verbose"
    ]
    
    # Add GPU specification (try 3 GPUs, will auto-adjust if fewer available)
    cmd.extend(["--gpus", "0,1,2"])
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("-" * 60)
    
    # Run the test
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        
        print("-" * 60)
        print()
        print(f"✓ Test completed successfully in {elapsed:.1f} seconds")
        print()
        
        # Check output files
        h5_files = [f for f in os.listdir(output_dir) if f.endswith('.h5')]
        print(f"Generated files ({len(h5_files)} total):")
        
        # Group by GPU
        gpu_files = {}
        for f in sorted(h5_files):
            if '_gpu' in f:
                gpu_id = f.split('_gpu')[1].split('_')[0]
                if gpu_id not in gpu_files:
                    gpu_files[gpu_id] = []
                gpu_files[gpu_id].append(f)
        
        for gpu_id, files in sorted(gpu_files.items()):
            print(f"  GPU {gpu_id}: {len(files)} files")
            for f in files[:3]:  # Show first 3
                print(f"    - {f}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
        
        print()
        print("✓ All tests passed!")
        print()
        print("You can now run full training with:")
        print("  python3 train_curriculum.py --selfplay-gpus 3 --use-cuda-mcts")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print()
        print(f"✗ Test failed with return code {e.returncode}")
        print()
        print("Common issues:")
        print("1. Not enough GPUs available (check with nvidia-smi)")
        print("2. CUDA not properly installed")
        print("3. Out of GPU memory (try reducing rollouts/threads)")
        print("4. Model file corrupted")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_single_gpu_fallback():
    """Test that single GPU fallback works"""
    print("\n" + "=" * 60)
    print("Testing Single GPU Fallback")
    print("=" * 60)
    print()
    
    cmd = [
        sys.executable,
        "create_training_games.py",
        "--model", "weights/AlphaZeroNet_20x256.pt",
        "--games-to-play", "2",
        "--rollouts", "5",
        "--threads", "2",
        "--save-format", "h5",
        "--output-dir", "test_single_gpu",
        "--file-base", "test_single",
        "--verbose"
    ]
    
    print("Testing standard single-GPU generation...")
    try:
        os.makedirs("test_single_gpu", exist_ok=True)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Single GPU generation works")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Single GPU generation failed: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False

if __name__ == "__main__":
    # Run tests
    success = True
    
    # Test single GPU first (baseline)
    if not test_single_gpu_fallback():
        print("\nSingle GPU test failed. Please fix this before testing multi-GPU.")
        sys.exit(1)
    
    # Test multi-GPU
    if not test_multigpu_generation():
        success = False
    
    sys.exit(0 if success else 1)