import numpy as np
def generate_gauss_mixture(k,R):
    n = 10000
    d = 15
    mean = np.zeros((15))
    cov = np.diag(np.array([R]*d))
    centers = np.random.multivariate_normal(mean, cov, k)

    for i in range(k):
        mean = centers[i]
        if i == 0:
            data = np.random.multivariate_normal(mean, np.diag(np.ones(d)), int(n/k+n%k))
            trueLabels = np.repeat(i,int(n/k+n%k))
        else:
            data = np.append(data, np.random.multivariate_normal(mean, np.diag(np.ones(d)) , int(n/k)), axis = 0) 
            trueLabels = np.append(trueLabels,np.repeat(i,int(n/k)))
    return data, trueLabels
    
generate_gauss_mixture(15,1)
