import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils.extmath import row_norms

class NaiveHartiganKMeans(BaseEstimator):
    """
    Scikit-learn compatible NaiveHartiganKMeans cluterer.
    """
    def __init__(self, n_clusters=3, random_state=None, init="kmeans++"):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.init = init

    def fit(self, X):
        x_squared_norms = row_norms(X, squared=True)
        rng = np.random.RandomState(self.random_state)

        if self.init == "kmeans++":
            # Private function of sklearn.cluster.k_means_, to get the initial centers.
            init_centers = _k_init(X, self.n_clusters, x_squared_norms, rng)
        elif self.init == "random":
            random_samples = rng.random_integers(0, X.shape[0], size=self.n_clusters)
            init_centers = X[random_samples, :]
        else:
            raise ValueError("init should be either kmeans++ or random")

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
            curr_label = label
            max_cost = np.inf
            while max_cost > 0:
                distances = x_squared_norms[i] - 2 * np.dot(sample, self.centers_.T) + np.sum(self.centers_**2, axis=1)

                curr_distance = distances[curr_label]
                other_distance = np.delete(distances, curr_label)
                curr_n_samples = self.n_samples_[curr_label]
                other_n_samples = np.delete(self.n_samples_, curr_label)
                cost = (curr_n_samples / (curr_n_samples - 1) * curr_distance) - (other_n_samples / (other_n_samples + 1) * other_distance)
                max_cost_ind = np.argmax(cost)
                max_cost = cost[max_cost_ind]

                if max_cost > 0:
                    # We deleted the label index from other_n_samples
                    if max_cost_ind > curr_label:
                        max_cost_ind += 1

                    # Reassign the clusters
                    self.labels_[i] = max_cost_ind

                    self.centers_[curr_label] = (curr_n_samples * self.centers_[curr_label] - sample) / (curr_n_samples - 1)
                    moved_n_samples = self.n_samples_[max_cost_ind]
                    self.centers_[max_cost_ind] = (moved_n_samples * self.centers_[max_cost_ind] + sample) / (moved_n_samples + 1)
                    self.n_samples_[curr_label] -= 1
                    self.n_samples_[max_cost_ind] += 1
                    curr_label = max_cost_ind
