import numpy as np
import Kmeans
from gauss_mixture import generate_gauss_mixture
from evaluate_cluster import cluster_cost, mis_class_rate
import pandas as pd

def gauss_mixture_evluate():
    k=50
    R=1
    totalCost = 0
    totalMisClass = 0
    totalIterations = 0
    for i in range(1):
        data, truelabels = generate_gauss_mixture(k, R)
        initial_centroids = data[np.random.choice(range(data.shape[0]), k, replace=False),:]
        iterations,centroids, labels = Kmeans.kmeans(data, k, initial_centroids)
        cost = cluster_cost(data, centroids)
        misclassrate = mis_class_rate(truelabels, labels)
        totalCost += cost
        totalIterations += iterations
        totalMisClass += misclassrate
    avgCost = totalCost
    avgMisClass = totalMisClass
    avgIterations = totalIterations
    print(avgCost)
    print(avgMisClass)
    print(avgIterations)

gauss_mixture_evluate()
def evaluate_spambase():
    f = open("spambase.data","r")
    df = pd.read_table('spambase.data', sep=',', names=range(58))
    df = np.array(df)