from sklearn.preprocessing import scale


def shift_to_mean_zero(data):
    return scale(data)
