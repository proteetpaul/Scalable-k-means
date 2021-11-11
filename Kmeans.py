def Kmeans(data, k, random_state):
    

    np.random.seed(random_state)
    centroids = []
    m = np.shape(data)[0]

    for _ in range(k):
        r = np.random.randint(0, m-1)
        centroids.append(data[r])

    return np.array(centroids)
