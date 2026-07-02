import zmq
import numpy as np
import pandas as pd
import time
import pickle
import uuid
import hashlib

print("=" * 60)
print("K-MEANS WORKER NODE (IRIS DATASET)")
print("=" * 60)

SERVER_IP = "192.168.137.1"  # Update if Laptop 1 IP changes
PORT_PUB = 5555
PORT_PULL = 5556
BATCH_SIZE = 2000

WORKER_ID = str(uuid.uuid4())
print(f"Worker ID: {WORKER_ID[:8]} | Server IP: {SERVER_IP}")

print("\nLoading Iris data...")
df = pd.read_csv('iris_augmented.csv')
X = df.values  # All 4 columns are features

# Hash-based partitioning
worker_hash = int(hashlib.md5(WORKER_ID.encode()).hexdigest()[:8], 16)
np.random.seed(worker_hash)
n_workers = 2
indices = np.random.permutation(len(X))
split_point = len(X) // n_workers

# Assign partition based on hash parity
X_local = X[indices[split_point:]] if worker_hash % 2 == 0 else X[indices[:split_point]]

print(f"Worker partition: {len(X_local):,} samples")
print(f"Processing 4 features per sample")

context = zmq.Context()
sub_socket = context.socket(zmq.SUB)
sub_socket.connect(f"tcp://{SERVER_IP}:{PORT_PUB}")
sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
push_socket = context.socket(zmq.PUSH)
push_socket.connect(f"tcp://{SERVER_IP}:{PORT_PULL}")

print(f"✅ Connected to server\n")
time.sleep(1)

for iteration in range(100):
    try:
        data = pickle.loads(sub_socket.recv())
        centers = data['centers']
    except: continue

    batch_idx = np.random.choice(len(X_local), min(BATCH_SIZE, len(X_local)), replace=False)
    X_batch = X_local[batch_idx]

    # Vectorized distance calculation for 4 features
    distances = np.linalg.norm(X_batch[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)

    new_centers = np.zeros_like(centers)
    counts = np.zeros(len(centers))
    for i in range(len(centers)):
        mask = assignments == i
        if np.any(mask):
            new_centers[i] = X_batch[mask].mean(axis=0)
            counts[i] = np.sum(mask)

    push_socket.send(pickle.dumps({
        'worker_id': WORKER_ID,
        'centers': new_centers,
        'counts': counts
    }))

    if iteration % 20 == 0: print(f"Iteration {iteration:3d} | Processed {len(X_batch)} samples")

print("\n✅ Worker finished training")
sub_socket.close()
push_socket.close()
context.term()
