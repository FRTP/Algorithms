import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def get_feature_importance(clf):
    if hasattr(clf, 'coef_'):
        return clf.coef_
    elif hasattr(clf, 'feature_importance_'):
        return clf.feature_importances_
    else:
        raise AttributeError()


def print_significant_features(pipeline=None, n=20):
    feature_names = pipeline.get_params()['vect'].get_feature_names()
    clf = pipeline.get_params()['clf']
    coefs = get_feature_importance(clf)
    print("Total features: {}".format(len(coefs)))
    coefs_with_fns = sorted(zip(coefs, feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_2, fn_2) in top:
        print("%.4f: %-16s" % (coef_2, str(fn_2)))


def plot_significant_features(pipeline=None, n=20, file_name=None):
    feature_names = pipeline.get_params()['vect'].get_feature_names()
    clf = pipeline.get_params()['clf']
    coefs = get_feature_importance(clf)

    print("Total features: {}".format(len(coefs)))
    coefs_with_fns = sorted(zip(coefs, feature_names))
    top = coefs_with_fns[:-(n + 1):-1]

    y, X = zip(*top)

    plt.figure(figsize=(15, 8))
    plt.title("Top 20 most important features")
    plt.gcf().subplots_adjust(bottom=0.25)
    ax = plt.subplot(111)

    ax.bar(range(len(X)), y, color="r", align="center")
    ax.set_xticks(range(len(X)))
    ax.set_xlim(-1, len(X))
    ax.set_xticklabels(X, rotation='vertical')

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()


def plot_confusion_matrix(y_true, y_pred, target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
