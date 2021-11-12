import numpy as np
from distance_func import distance

def kmeans(data, k, initial_centroids, iterations=10000):
    centroids = initial_centroids
    m = np.shape(data)[0]
    cluster_label = np.zeros((m))
    iter = 10000

    for i in range(iterations):
        dist = distance(data, centroids)
        cluster_label = np.argmin(dist, axis=1)
        newCentroids = np.zeros(centroids.shape)
        for j in range(0,k):
            if sum(cluster_label == j) == 0:
                newCentroids[j] = centroids[j]
            else:
                newCentroids[j] = np.mean(data[cluster_label == j, :], axis=0)
        if np.array_equal(centroids, newCentroids):
            print("Convergence reached after:",i," iterations...")
            iter = i
            break
        centroids = newCentroids
    
    return iter, centroids, cluster_label
