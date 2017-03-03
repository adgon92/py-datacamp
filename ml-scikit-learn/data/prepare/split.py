from sklearn.model_selection import train_test_split


def split_data_set(data, digits):
    X_train, Y_train, y_train, y_test, images_train, images_test = train_test_split(
        data,
        digits.target,
        digits.images,
        test_size=0.25,
        random_state=42
    )
    return X_train, Y_train, y_test, images_train, images_test
