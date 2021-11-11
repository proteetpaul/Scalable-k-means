import numpy as np
import math
from distance_func import distance
from kmeanspp_func import cost, distribution, sample_new

def get_weight(dist, centroids):
    return np.sum(np.min())

def ScalableKMeansPlusPlus(data, k, l):
    centroids = data[np.random.choice(range(data.shape[0]),1), :]
    dist = distance(data, centroids)
    initial_cost = np.sum(dist)
    iter = math.log(initial_cost)

    for i in range(0, iter):
        dist = distance(data, centroids)
        current_cost = cost(dist)
        distrib = distribution(dist, current_cost)
        centroids = np.r_[centroids, sample_new(data,distrib,l)]
    
    final_centroids = ScalableKMeansPlusPlus(centroids, k)
    return final_centroids
