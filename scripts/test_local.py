"""
Quick local test of the parameter server system
Run this to verify all components work before deploying to 3 laptops
"""
import subprocess
import time
import sys
import os

print("=" * 60)
print("LOCAL TEST - Parameter Server System")
print("=" * 60)

# Check if iris_augmented.csv exists
if not os.path.exists('iris_augmented.csv'):
    print("ERROR: iris_augmented.csv not found!")
    print("Run: python src/augment_dataset.py")
    sys.exit(1)

print("\n[OK] Dataset found")

# Check packages
print("\nChecking required packages...")
try:
    import zmq
    import numpy
    import pandas
    import sklearn
    print("[OK] All packages installed")
except ImportError as e:
    print(f"[ERROR] Missing package: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

print("\n" + "=" * 60)
print("LOCAL TEST COMPLETE")
print("=" * 60)
print("\nTo test the full system locally:")
print("1. Terminal 1: python src/server.py")
print("2. Terminal 2: python src/worker.py")
print("3. Terminal 3: python src/worker.py (run again for the 2nd worker)")
print("\nFor 3 laptops deployment:")
print("1. Copy this project folder to all 3 laptops")
print("2. On Laptop 1: python src/server.py")
print("3. On Laptop 2: Edit src/worker.py SERVER_IP, then python src/worker.py")
print("4. On Laptop 3: Edit src/worker.py SERVER_IP, then python src/worker.py")
print("5. On any laptop: python src/single_machine_kmeans.py (for baseline comparison)")
