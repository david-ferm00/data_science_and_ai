# Codegrade Tag Question1
# Do *not* remove the tag above
# Implement the class below

import numpy as np

class KMeansManual:
    """
    Implements Lloyd's algorithm for k-means clustering.

    Member variables:
    - n_clusters : int, the parameter k (number of clusters)
    - init : array, optional, initial clusters (or none)
    - n_feature_in_ : int, length of vectors
    - labels_ : array of integers of length n, after fitting, the cluster labels of the dataset
    - cluster_centers_ : array of shape n*d, the fitted cluster centers
    - inertia_ : float, sum of squared euclidean distances from each datapoint to the associated cluster center
    """

    def __init__(self, n_clusters, init = None, random_state = None):
        """
        Constructor, sets the parameters

        Parameters:
        - n_clusters : int, the parameter k (number of clusters)
        - init : array, optional, initial clusters; if None, clusters will be randomly initialized by selecting k random points from the dataset during fit(); otherwise, must be an array of shape k*d and the number of features in the dataset must match d
        - random state : integer, optional, if init is not None, this parameter is ignored; sets the random number seed for selecting the random datapoints
        """
        assert n_clusters > 0
        self.n_clusters = n_clusters
        self.random_state = random_state
        if init is not None:
            assert init.ndim == 2
            assert init.shape[0] == n_clusters
            self.init = init.copy()
        else:
            self.init = None

    def assign_labels(self, X, cluster_centers):
        """
        Given the n*d dataset X, returns an integer array of length n where each element has value 0, 1, ..., k-1, to indicate the index of the closest cluster center (in terms of squared Euclidean distance)

        Parameters:
        - X : array, n*d dataset
        - cluster_centers : array of shape k*d, present cluster centers

        Return value:
        - an integer array of length n where the ith element corresponds to the cluster label of the closest cluster center of that datapoint
        """
        n = X.shape[0]

        k = cluster_centers.shape[0]

        labels = []

        for i in range(n):
            distance = []
            for j in range(k):
                dist = np.sum((X[i] - cluster_centers[j])**2)
                distance.append(dist)

            labels.append(np.argmin(distance))

        

        return labels
    def compute_cluster_centers(self, X, labels):
        """
        Determine the cluster centroids, that is, the mean of the vectors associated with the same cluster label.

        Parameters:
        - X : array of shape n*d, the dataset
        - labels : 1-dimensional integer array of length n, the present label assignments

        Return value:
        - An array of shape k*d, the current cluster centers
        """
        assert isinstance(X,np.ndarray)
        assert isinstance(labels,np.ndarray)
        assert X.ndim == 2
        assert labels.ndim == 1
        assert X.shape[0] == labels.shape[0]
        raise NotImplementedError

    def compute_inertia(self, X, labels, cluster_centers):
        """
        Given the n*d dataset, the n-vector of cluster labels, and the k*d array of cluster centers, computes the inertia, that is, the sum of squared Euclidean distances from each datapoint to the closest cluster center (as identified by the cluster label)

        Parameters:
        - X : array of shape n*d, the dataset
        - labels : integer array of length n, the cluster labels of each datapoint
        - cluster_centers : array of shape k*d, the present cluster centers

        Return value:
        - float, the sum of squared euclidean distances to closest cluster centers
        """
        assert isinstance(X,np.ndarray)
        assert isinstance(cluster_centers,np.ndarray)
        assert isinstance(labels,np.ndarray)
        assert X.ndim == 2 and cluster_centers.ndim == 2
        assert labels.ndim == 1
        assert cluster_centers.shape[0] == self.n_clusters
        assert X.shape[0] == labels.shape[0]
        assert X.shape[1] == cluster_centers.shape[1]
        raise NotImplementedError

    def fit(self,X):
        """
        Fit the dataset X, that is, compute the cluster centers and assignments.

        Parameters:
        - X : n*d array of observations; if init was not None, then d must match the second axis of init; otherwise, k random row vectors are picked from X to serve as initial clusters (if random_state was not None, then the same choice should always be made)

        Return value:
        - Returns self
        """
        raise NotImplementedError