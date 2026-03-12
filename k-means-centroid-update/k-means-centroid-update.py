"""
## K-Means Centroid Update

After assigning each point to a cluster, the centroid update step recomputes each centroid as the mean of all points assigned to it. This is the second half of one K-Means iteration.

Given a list of data points, their cluster assignments, and the number of clusters k, compute the new centroid positions.

### Formula

For each cluster j, the new centroid is the mean of all assigned points:

$$c_j=\frac{1}{|S_j|}\sum_{p\in S_j}p$$

Where $S_j$ is the set of points assigned to cluster $j$.

### Examples

Input: points = [[0, 0], [2, 2], [10, 10], [12, 12]], assignments = [0, 0, 1, 1], k = 2
Output: [[1.0, 1.0], [11.0, 11.0]]
Cluster 0 contains [0,0] and [2,2], mean = [1,1]. Cluster 1 contains [10,10] and [12,12], mean = [11,11].

Input: points = [[0, 0], [1, 0], [5, 5], [6, 5], [10, 0]], assignments = [0, 0, 1, 1, 2], k = 3

Output: [[0.5, 0.0], [5.5, 5.0], [10.0, 0.0]]

Each cluster's centroid is the element-wise mean of its assigned points.

### Hints

Hint 1: Create a list of k zero vectors (one per cluster) and a count array. Loop through points, adding each to its assigned cluster's sum and incrementing the count. Then divide each sum by its count.

Hint 2: Be careful with empty clusters (count = 0). Check the count before dividing to avoid division by zero.

### Requirements

- For each cluster, compute the mean of all assigned points along each dimension
- If a cluster has no assigned points, return a zero vector for that centroid
- Return a list of k centroids, each a list of floats

### Constraints

- assignments[i] is an integer in [0, k-1]
- All points have the same dimensionality
- Return a list of k centroids, each a list of floats
- Time limit: 300 ms
"""

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    if not points:
        return []

    dim = len(points[0])

    sums = [[0.0] * dim for _ in range(k)]
    counts = [0] * k

    for point, cluster in zip(points, assignments):
        counts[cluster] += 1
        for d in range(dim):
            sums[cluster][d] += point[d]

    centroids = []
    for cluster in range(k):
        if counts[cluster] == 0:
            centroids.append([0.0] * dim)
        else:
            centroids.append([
                sums[cluster][d] / counts[cluster]
                for d in range(dim)
            ])

    return centroids