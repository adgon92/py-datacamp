import pandas as pd

if __name__ == "__main__":
    training_set = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
        header=None
    )

    test_set = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",
        header=None
    )

    print(training_set)
    print(test_set)
