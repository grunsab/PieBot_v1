#!/usr/bin/env python3
"""
Test simple threading issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time

class TestWorker(threading.Thread):
    def __init__(self, worker_id):
        super().__init__(daemon=False)
        self.worker_id = worker_id
        print(f"Worker {worker_id} created")
    
    def run(self):
        print(f"Worker {self.worker_id}: Starting")
        time.sleep(0.1)
        print(f"Worker {self.worker_id}: Ending")
        # Run method ends here

print("Creating worker...")
worker = TestWorker(0)

print("Starting worker...")
worker.start()

print("Waiting for worker (1 second timeout)...")
worker.join(timeout=1.0)

if worker.is_alive():
    print("ERROR: Worker still alive!")
else:
    print("SUCCESS: Worker terminated properly")

print("Done")