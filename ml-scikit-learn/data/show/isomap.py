from sklearn.manifold import Isomap
from matplotlib import pyplot as plt


class IsoMap:

    def __init__(self, x_data):
        self._x_data = x_data
        self._x_iso = Isomap(n_neighbors=10).fit_transform(x_data)

    def show_clustered_data(self, y_train, clusters):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.85)

        ax[0].scatter(self._x_data[:, 0], self._x_data[:, 1], c=clusters)
        ax[0].set_title('Predicted Training Labels')
        ax[1].scatter(self._x_data[:, 0], self._x_data[:, 1], c=y_train)
        ax[1].set_title('Actual Training Labels')

        plt.show()
