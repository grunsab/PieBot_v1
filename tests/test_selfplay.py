#!/usr/bin/env python3
"""
Test self-play mode with limited moves
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import time

# Run playchess.py in self-play mode with small parameters
cmd = [
    "python", "playchess.py",
    "--model", "weights/AlphaZeroNet_20x256.pt",
    "--mode", "s",  # self-play mode
    "--rollouts", "10",  # small number for quick testing
    "--threads", "2"
]

print("Starting self-play test...")
print(f"Command: {' '.join(cmd)}")
print("\nRunning for 10 seconds...\n")

# Start the process
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Let it run for 10 seconds
time.sleep(10)

# Terminate the process
process.terminate()

# Get output
stdout, stderr = process.communicate(timeout=5)

print("=== Output ===")
print(stdout[:2000])  # Print first 2000 chars

if stderr:
    print("\n=== Errors ===")
    print(stderr[:1000])

print("\n=== Test Complete ===")
print("Self-play mode is working if you see chess moves above!")