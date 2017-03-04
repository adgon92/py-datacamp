from sklearn import cluster


class KMeansCluster:

    _INIT = "k-means++"
    _N_CLUSTERS = 10
    _RANDOM_STATE = 42

    def __init__(self, training_data):
        self._training_data = training_data
        self._clf = cluster.KMeans(
            init=KMeansCluster._INIT,
            n_clusters=KMeansCluster._N_CLUSTERS,
            random_state=KMeansCluster._RANDOM_STATE
        )

    @property
    def cluster_centers(self):
        return self._clf.cluster_centers_

    def fit_training_set(self):
        self._clf.fit(self._training_data)

    def predict_labels(self):
        self._clf.predict(self._training_data)

    def compute_cluster_centres_and_predict_cluster_index(self):
        return self._clf.fit_predict(self._training_data)
