import numpy as np
from kmeanspp_func import cost
from distance_func import distance
import pandas as pd

def cluster_cost(data, centroids):
    dist = distance(data,centroids)
    return cost(dist)/(10**4)

def mis_class_rate(trueLabels, labels):
    # print(np.shape(trueLabels))
    # print(np.shape(labels))
    # n = np.shape(labels)[0]
    # misclass = 0
    # for i in range(0,n):
    #     if trueLabels[i] != labels[i]:
    #         misclass += 1
    # return misclass/n
    n = np.shape(labels)[0]
    df = pd.DataFrame({'True':trueLabels, 'Predict':labels,'V':1})
    table = pd.pivot_table(df, values ='V', index = ['True'], columns=['Predict'], aggfunc=np.sum).fillna(0)
    misRate = 1-sum(table.max(axis=1))/n
    return misRate

