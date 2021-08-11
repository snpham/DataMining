from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


if __name__ == '__main__':
    
    pass

    x = np.array([10 , 7, 6, 3, 28, 26, 22, 28])
    kmeans = KMeans(n_clusters=2)
    kmeans = kmeans.fit(x.reshape(-1,1))
    print(kmeans.cluster_centers_)