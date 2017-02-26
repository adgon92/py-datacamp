
from data.load.sklearn.dataset import get_digits
from sklearn.decomposition import PCA


class PCAExample:

    RANDOMIZED_PCA = PCA(n_components=2, svd_solver='randomized')
    REGULAR_PCA = PCA(n_components=2)

    def __init__(self, data):
        self._data = data

    def get_randomized_pca(self):
        return self.RANDOMIZED_PCA.fit_transform(self._data)

    def show_randomized_pca_result(self):
        print("Randomized PCA:\n", self.get_randomized_pca())

    def show_regular_pca_result(self):
        regular_pca_result = self.REGULAR_PCA.fit_transform(self._data)
        print("Regular PCA:\n", regular_pca_result)


if __name__ == "__main__":
    digits = get_digits()
    pca = PCAExample(digits.data)
    pca.show_randomized_pca_result()
    pca.show_regular_pca_result()
