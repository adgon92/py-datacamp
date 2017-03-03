from data.clustering.kmeans import KMeansCluster
from data.load.sklearn.dataset import get_digits
from data.prepare.normalization import shift_to_mean_zero
from data.prepare.split import split_data_set


def main():
    digits = get_digits()
    data = shift_to_mean_zero(data=digits.data)
    # noinspection PyPep8Naming
    X_train, Y_train, y_test, images_train, images_test = split_data_set(
        data=data,
        digits=digits
    )
    cluster = KMeansCluster(X_train)
    cluster.fit_training_set()


if __name__ == "__main__":
    main()