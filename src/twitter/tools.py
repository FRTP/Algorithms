import matplotlib.pyplot as plt


def print_significant_features(pipeline=None, n=20):
    feature_names = pipeline.get_params()['vect'].get_feature_names()
    coefs = []
    try:
        coefs = pipeline.get_params()['clf'].coef_
    except:
        coefs.append(pipeline.get_params()['clf'].feature_importances_)
    print("Total features: {}".format(len(coefs[0])))
    coefs_with_fns = sorted(zip(coefs[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_2, fn_2) in top:
        print("%.4f: %-16s" % (coef_2, str(fn_2)))


def plot_significant_features(pipeline=None, n=20, file_name=None):
    feature_names = pipeline.get_params()['vect'].get_feature_names()
    coefs = []
    try:
        coefs = pipeline.get_params()['clf'].coef_
    except:
        coefs.append(pipeline.get_params()['clf'].feature_importances_)

    print("Total features: {}".format(len(coefs[0])))
    coefs_with_fns = sorted(zip(coefs[0], feature_names))
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
