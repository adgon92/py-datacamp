from data.clustering.kmeans import KMeansCluster
from data.load.sklearn.dataset import get_digits
from data.prepare.normalization import shift_to_mean_zero
from data.prepare.pca import PCAExample
from data.prepare.split import split_data_set
from data.show.isomap import IsoMap


def main():
    digits = get_digits()
    data = shift_to_mean_zero(data=digits.data)
    # noinspection PyPep8Naming
    X_train, Y_train, y_train, y_test, images_train, images_test = split_data_set(
        data=data,
        digits=digits
    )

    # noinspection PyPep8Naming
    X_pca = PCAExample(X_train).get_randomized_pca()

    clusters = KMeansCluster(X_train)\
        .compute_cluster_centres_and_predict_cluster_index()

    iso_map = IsoMap(X_pca)
    iso_map.show_clustered_data(y_train, clusters)

if __name__ == "__main__":
    main()