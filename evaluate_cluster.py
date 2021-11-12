import numpy as np
from kmeanspp_func import cost
from distance_func import distance

def cluster_cost(data, centroids):
    dist = distance(data,centroids)
    return cost(dist)/(10**4)

def mis_class_rate(trueLabels, labels):
    n = np.shape(labels)[0]
    misclass = 0
    for i in range(0,n):
        if trueLabels[i] != labels[i]:
            misclass += 1
    return misclass/n

