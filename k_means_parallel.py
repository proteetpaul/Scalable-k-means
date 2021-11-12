import numpy as np
import math
from distance_func import distance
from kmeanspp_func import cost, distribution, sample_new

def get_weight(dist,centroids):
    min_dist = np.zeros(dist.shape)
    min_dist[range(dist.shape[0]), np.argmin(dist, axis=1)] = 1
    count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(centroids.shape[0])])
    return count/np.sum(count)

def ScalableKMeansPlusPlus(data, k, l, iter=5):
    centroids = data[np.random.choice(range(data.shape[0]),1), :]
    dist = distance(data, centroids)
    initial_cost = np.sum(dist)

    for i in range(0, iter):
        dist = distance(data, centroids)
        current_cost = cost(dist)
        distrib = distribution(dist, current_cost)
        centroids = np.r_[centroids, sample_new(data,distrib,l)]
    
    ## reduce k*l to k using KMeans++ 
    dist = distance(data, centroids)
    weights = get_weight(dist, centroids)
    
    return centroids[np.random.choice(len(weights), k, replace= False, p = weights),:]
