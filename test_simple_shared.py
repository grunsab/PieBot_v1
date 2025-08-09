#!/usr/bin/env python3
"""
Simple test to check if multiprocessing works correctly.
"""

import multiprocessing
from multiprocessing import Pool, Manager
import sys

print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

# Test basic multiprocessing
def worker_func(x):
    return x * 2

def test_basic():
    print("\nTest 1: Basic multiprocessing")
    try:
        with Pool(2) as pool:
            results = pool.map(worker_func, [1, 2, 3, 4])
            print(f"Results: {results}")
            print("Basic multiprocessing works!")
    except Exception as e:
        print(f"Basic multiprocessing failed: {e}")

# Test Manager
def test_manager():
    print("\nTest 2: Manager dict and lock")
    try:
        manager = Manager()
        shared_dict = manager.dict()
        shared_lock = manager.Lock()
        
        shared_dict['test'] = 123
        print(f"Shared dict works: {shared_dict['test']}")
        
        with shared_lock:
            print("Lock works!")
            
        print("Manager works!")
    except Exception as e:
        print(f"Manager failed: {e}")

# Test passing Manager objects to workers
worker_dict = None
worker_lock = None

def init_worker(d, l):
    global worker_dict, worker_lock
    worker_dict = d
    worker_lock = l

def worker_with_shared(x):
    global worker_dict, worker_lock
    with worker_lock:
        worker_dict[x] = x * 2
    return x

def test_shared():
    print("\nTest 3: Passing Manager objects to workers")
    try:
        manager = Manager()
        shared_dict = manager.dict()
        shared_lock = manager.Lock()
        
        with Pool(2, initializer=init_worker, initargs=(shared_dict, shared_lock)) as pool:
            results = pool.map(worker_with_shared, [1, 2, 3, 4])
            print(f"Worker results: {results}")
            print(f"Shared dict contents: {dict(shared_dict)}")
            print("Shared objects in workers work!")
    except Exception as e:
        print(f"Shared objects in workers failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing multiprocessing setup...")
    test_basic()
    test_manager()
    test_shared()
    print("\nAll tests completed!")