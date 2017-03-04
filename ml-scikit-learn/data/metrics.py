from sklearn import metrics
from sklearn.metrics import (
    homogeneity_score, completeness_score, v_measure_score,
    adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
)


def show_confusion_matrix(y_test, y_pred):
    print(metrics.confusion_matrix(y_test, y_pred))


def show_metrics(clf, X_test, y_test, y_pred):
    score = silhouette_score(X_test, y_pred, metric='euclidean')
    print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
    print(f'{clf.interia_}   '
          f'{homogeneity_score(y_test, y_pred):.3f}   '
          f'{completeness_score(y_test, y_pred):.3f}   '
          f'{v_measure_score(y_test, y_pred):.3f}   '
          f'{adjusted_rand_score(y_test, y_pred):.3f}   '
          f'{adjusted_mutual_info_score(y_test, y_pred):.3f}   '
          f'{score}'
          )
