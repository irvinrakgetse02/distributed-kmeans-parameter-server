# Results

All numbers below are from actual runs — either the team's real 3-laptop deployment, or reproduced directly from the shipped code in this repo.

## Setup
- Dataset: Iris dataset (4 features), augmented to 150,000 samples
- Clusters: k = 3
- Distributed setup: 1 parameter server + 2 worker nodes across 3 laptops on a local Wi-Fi hotspot (ZeroMQ PUB/SUB + PUSH/PULL)

## Measured performance

| Metric | Single Machine (scikit-learn) | Distributed (parameter server, 3 laptops) |
|---|---|---|
| Wall-clock time | 9.4s | 20.0s |
| Final inertia | 78,851.44 | 78,856.89 (0.007% difference) |

Timing figures are from the team's real 3-laptop run. The distributed version is about 2x slower — expected at this dataset size, since network round-trips between server and workers cost more than the compute they save. This is a direct, measured illustration of Amdahl's Law: distributing pays off once compute cost dominates communication cost, and 150K rows of 4-dimensional K-Means is too light a workload for that to be true here.

## A real bug we found and fixed: initialization matters more than the network

Our first working version of the parameter server initialized cluster centroids with small random values (`np.random.randn(...) * 2`), unrelated to the actual scale of the data. Running the exact aggregation logic from `server.py`/`worker.py` end-to-end, this converged to a **final inertia of ~681,400** — about 8.6x worse than the single-machine baseline's 78,851. The distributed run technically "worked" (it did converge, and it did drop inertia fast), but it settled into a much worse local optimum than scikit-learn's `k-means++`-initialized run.

The fix: seed the initial centroids from real sampled data points (a simple k-means++-style seeding) instead of arbitrary small values, and use a larger local batch size per worker (2,000 vs 500) for more stable update estimates. With that change, the distributed run converges to **78,856.89** — a 0.007% difference from the single-machine baseline. See `src/kmeans_parameters.py` for the fix and `results/convergence_log.csv` for the logged trace.

This is worth including here rather than glossing over: it's a genuine distributed-systems lesson — a synchronization strategy can be implemented correctly and still produce a much worse answer if the thing being synchronized starts in a bad state. Good initialization strategy matters as much as the communication pattern.

## Visualizations

- `performance_comparison.png` — wall-clock time, single vs. distributed
- `quality_comparison.png` — final inertia, single vs. distributed
- `convergence_plot.png` — real logged convergence trace of the distributed run
- `single_machine_clusters.png` — single-machine K-Means result on the raw Iris features
- `distributed_clusters.png` — the dataset colored by the distributed run's final cluster assignments, with the saved centroids from `final_cluster_centers.npy`

## Honest limitations

- Timing numbers are from one measured run each on the team's hardware, not averaged over multiple trials.
- Tested at one scale (150K rows, 3 nodes, k=3). The compute/communication tradeoff would look different for a heavier workload (e.g. a neural network parameter server), where distributed training is expected to actually pay off in wall-clock time as well as quality.
