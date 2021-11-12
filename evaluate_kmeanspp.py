import numpy as np
import Kmeans
from gauss_mixture import generate_gauss_mixture
from evaluate_cluster import cluster_cost, mis_class_rate
import kmeanspp_func
from distance_func import distance
import pandas as pd

def evaluate_gauss():
    k=50
    R=10
    totalCost = 0
    totalMisClass = 0
    totalIterations = 0
    totalInitialCost = 0
    for i in range(1):
        data, truelabels = generate_gauss_mixture(k, R)
        initial_centroids = kmeanspp_func.Kmeanspp(data, k)
        dist = distance(data, initial_centroids)
        initial_cost = kmeanspp_func.cost(dist)/(10**4)
        totalInitialCost += initial_cost

        iterations,centroids, labels = Kmeans.kmeans(data, k, initial_centroids)
        cost = cluster_cost(data, centroids)
        misclassrate = mis_class_rate(truelabels, labels)
        totalCost += cost
        totalIterations += iterations
        totalMisClass += misclassrate

    avgCost = totalCost
    avgMisClass = totalMisClass
    avgIterations = totalIterations
    avgInitialCost = totalInitialCost
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
    initial_centroids = kmeanspp_func.Kmeanspp(df, k)
    dist = distance(df, initial_centroids)
    initial_cost = kmeanspp_func.cost(dist)/(10**4)

    iterations,centroids, labels = Kmeans.kmeans(df, k, initial_centroids)
    cost = cluster_cost(df, centroids)
    print(cost)
    print(iterations)
    print(initial_cost)
evaluate_spambase()
    
