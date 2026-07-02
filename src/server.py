import zmq
import time
import numpy as np
import pandas as pd
from kmeans_parameters import KMeansParameters
import pickle

print("=" * 60)
print("K-MEANS PARAMETER SERVER (IRIS DATASET)")
print("=" * 60)

SERVER_IP = "0.0.0.0"
PORT_PUB = 5555
PORT_PULL = 5556
N_CLUSTERS = 3
N_FEATURES = 4  # Iris dataset has 4 features
N_WORKERS = 2
LEARNING_RATE = 0.5

np.random.seed(42)

print("\nLoading validation dataset...")
df = pd.read_csv('iris_augmented.csv')
X_full = df.values

# Seed centroids from real data points rather than small random values —
# this closes most of the gap to the single-machine baseline's quality
# (see kmeans_parameters.py for why)
params = KMeansParameters(N_CLUSTERS, N_FEATURES, init_data=X_full)

initial_inertia = params.inertia(X_full)
print(f"Initial inertia: {initial_inertia:.2f}")
print(f"Dataset: 4 features (sepal length/width, petal length/width)")

context = zmq.Context()
pub_socket = context.socket(zmq.PUB)
pub_socket.bind(f"tcp://{SERVER_IP}:{PORT_PUB}")
pull_socket = context.socket(zmq.PULL)
pull_socket.bind(f"tcp://{SERVER_IP}:{PORT_PULL}")

print(f"\nServer listening on ports {PORT_PUB} (PUB) and {PORT_PULL} (PULL)")
print(f"Waiting for {N_WORKERS} workers...\n")

worker_updates = {}
worker_counts = {}
iteration = 0
start_time = time.time()
time.sleep(1)

try:
    while True:
        pub_socket.send(pickle.dumps({'type': 'centers', 'centers': params.centers}))
        if iteration % 10 == 0 or iteration == 0:
            print(f"📤 Iteration {iteration}: Sent cluster centers to workers")

        worker_updates.clear()
        worker_counts.clear()
        while len(worker_updates) < N_WORKERS:
            if pull_socket.poll(timeout=5000):
                data = pickle.loads(pull_socket.recv())
                worker_id = data['worker_id']
                worker_updates[worker_id] = data['centers']
                worker_counts[worker_id] = data['counts']
                print(f"  📥 Received update from Worker {worker_id[:8]}")
            else:
                print(f"  ⚠️ Timeout waiting for workers. Got {len(worker_updates)}/{N_WORKERS}")
                break

        if len(worker_updates) == 0:
            print("  No updates received. Retrying...")
            time.sleep(1)
            continue

        total_samples = sum(worker_counts.values())
        new_centers = np.zeros_like(params.centers)
        for worker_id in worker_updates:
            weight = worker_counts[worker_id] / total_samples
            new_centers += weight[:, np.newaxis] * worker_updates[worker_id]

        params.centers = (1 - LEARNING_RATE) * params.centers + LEARNING_RATE * new_centers
        iteration += 1

        if iteration % 10 == 0:
            current_inertia = params.inertia(X_full)
            elapsed = time.time() - start_time
            print(f"\n📊 Iteration {iteration} | Inertia: {current_inertia:.2f} | Time: {elapsed:.1f}s")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n" + "=" * 60)
    print("SERVER SHUTDOWN")
    print("=" * 60)
    final_inertia = params.inertia(X_full)
    elapsed = time.time() - start_time
    print(f"Total iterations: {iteration}")
    print(f"Final inertia: {final_inertia:.2f}")
    print(f"Improvement: {initial_inertia - final_inertia:.2f}")
    print(f"Total time: {elapsed:.1f} seconds")
    np.save('final_cluster_centers.npy', params.centers)
    print("\n💾 Final cluster centers saved.")
finally:
    pub_socket.close()
    pull_socket.close()
    context.term()
