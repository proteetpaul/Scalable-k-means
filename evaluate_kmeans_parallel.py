import numpy as np
import Kmeans
from gauss_mixture import generate_gauss_mixture
from evaluate_cluster import cluster_cost, mis_class_rate
import kmeanspp_func
from distance_func import distance
from k_means_parallel import scalableKMeansPlusPlus
import pandas as pd

def evaluate_gauss():
    k=50
    R=1
    l = 100
    iter = 5
    totalCost = 0
    totalMisClass = 0
    totalIterations = 0
    totalInitialCost = 0
    for i in range(1):
        data, truelabels = generate_gauss_mixture(k, R)
        initial_centroids = scalableKMeansPlusPlus(data, k, l, iter)
        initial_cost = cluster_cost(data, initial_centroids)
        totalInitialCost += initial_cost

        iterations,centroids, labels = Kmeans.kmeans(data, k, initial_centroids)
        cost = cluster_cost(data, centroids)
        misclassrate = mis_class_rate(truelabels, labels)
        totalCost += cost
        totalIterations += iterations
        totalMisClass += misclassrate

    avgCost = totalCost#/20
    avgMisClass = totalMisClass#/20
    avgIterations = totalIterations#/20
    avgInitialCost = totalInitialCost#/20
    print(avgCost)
    print(avgMisClass)
    print(avgIterations)
    print(avgInitialCost)
# evaluate_gauss()

def evaluate_spambase():
    f = open("spambase.data","r")
    df = pd.read_table('spambase.data', sep=',', names=range(58))
    df = np.array(df)
    k=20
    l = 10
    iter = 5
    initial_centroids = scalableKMeansPlusPlus(df, k, l, iter)
    dist = distance(df, initial_centroids)
    initial_cost = kmeanspp_func.cost(dist)/(10**4)

    iterations,centroids, labels = Kmeans.kmeans(df, k, initial_centroids)
    cost = cluster_cost(df, centroids)
    print(cost)
    print(iterations)
    print(initial_cost)
evaluate_spambase()