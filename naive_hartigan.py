import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils.extmath import row_norms

class NaiveHartiganKMeans(BaseEstimator):
	"""
	Scikit-learn compatible NaiveHartiganKMeans cluterer.
	"""

	def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

	def fit(self, X):
        x_squared_norms = row_norms(X, squared=True)
        rng = np.random.RandomState(0)

        # Private function of sklearn.cluster.k_means_, to get the initial centers.
        init_centers = _k_init(X, self.n_clusters, x_squared_norms, rng)

        # Assign initial labels. skip norm of x**2
        init_distances = np.sum(init_centers**2, axis=1) - 2 * np.dot(X, init_centers.T)
        init_labels = np.argmin(init_distances, axis=1)
        self.labels_ = init_labels

        self.centers_ = init_centers
        self.n_samples_ = np.zeros(self.n_clusters)

        # Count the number of samples in each cluster.
        for i in range(self.n_clusters):
            self.n_samples_[i] = np.sum(self.labels_ == i)

        for i, (sample, label) in enumerate(zip(X, self.labels_)):
            present_n_sample = self.n_samples_[label]
            # present_cost = present_n_sample * init_distances[i][label] / (present_n_sample - 1)

            distances = -2 * np.dot(sample, self.centers_.T) + np.sum(self.centers_**2, axis=1)
            curr_distance = distances[label]
            other_distance = np.delete(distances, label)
            curr_n_samples = self.n_samples_[label]
            other_n_samples = np.delete(self.n_samples_, label)
            cost = (curr_n_samples / (curr_n_samples - 1) * curr_distance) - (other_n_samples / (other_n_samples + 1) * other_distance)
            # print(np.max(cost))

