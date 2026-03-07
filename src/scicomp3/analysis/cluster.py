"""
Analysis of the DLA cluster growth
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ClusterStats:
    """
    Container for statistics of a growth cluster

    Attributes:
        highest_point: The y coordinate of the highest point
            that is part of the cluster
        broadness: The distance from the leftmost point to the rightmost
            point of the cluster
        fractal_dimension: The fractal dimension of the cluster
    """

    highest_point: int
    broadness: int
    fractal_dimension: float


def compute_cluster_stats(growth_mask: np.ndarray) -> ClusterStats:
    """Compute the statistics from a given growth_mask"""
    coords = np.argwhere(growth_mask)  # shape (k, 2): all cluster points

    # Highest point: maximum y (row) index in the cluster
    highest_point = int(np.max(coords[:, 1]))

    # Broadness: distance from leftmost to rightmost x (column) index
    broadness = int(np.max(coords[:, 0]) - np.min(coords[:, 0]))

    # Fractal dimension via box-counting
    N = growth_mask.shape[0] - 1
    box_sizes = [2**k for k in range(1, int(np.log2(N)))]
    box_counts = []
    for box_size in box_sizes:
        # Count how many boxes of this size contain at least one cluster point
        boxes_occupied = set((i // box_size, j // box_size) for i, j in coords)
        box_counts.append(len(boxes_occupied))

    # Fractal dimension is the slope of log(count) vs log(1/size)
    log_sizes = np.log(1.0 / np.array(box_sizes))
    log_counts = np.log(np.array(box_counts))
    fractal_dim = float(np.polyfit(log_sizes, log_counts, 1)[0])

    return ClusterStats(highest_point, broadness, fractal_dim)
