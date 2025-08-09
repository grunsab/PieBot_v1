#!/usr/bin/env python3
"""
Test if threading.Thread inheritance works properly on Windows
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time

print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

# Method 1: Thread inheritance
class WorkerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        
    def run(self):
        print("WorkerThread: Starting")
        time.sleep(0.1)
        print("WorkerThread: Ending")

# Method 2: Function with Thread
def worker_func():
    print("worker_func: Starting")
    time.sleep(0.1)
    print("worker_func: Ending")

print("\n=== Test 1: Thread inheritance ===")
t1 = WorkerThread()
t1.start()
print(f"Before join: is_alive={t1.is_alive()}")
t1.join(timeout=1.0)
print(f"After join: is_alive={t1.is_alive()}")

print("\n=== Test 2: Function with Thread ===")
t2 = threading.Thread(target=worker_func, daemon=True)
t2.start()
print(f"Before join: is_alive={t2.is_alive()}")
t2.join(timeout=1.0)
print(f"After join: is_alive={t2.is_alive()}")

print("\nDone!")