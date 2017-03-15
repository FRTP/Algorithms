import pandas as pd

from time import time
from sklearn.model_selection import train_test_split, cross_val_score

from tools.tools import plot_significant_features

# from models.baseline_model import build_model
from models.baseline_model import build_model

import logging


def main():
    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv('data/Tweets.csv')
    X = df['text'].values
    y = df['airline_sentiment']
    y = y.map({'negative': -1, 'neutral': 0, 'positive': 1}).values

    model = build_model()

    t0 = time()

    scores = cross_val_score(model, X, y)
    score = scores.mean()
    print("Score:{}".format(score))
    # plot_significant_features(pipeline=model,
    #                           file_name="sentiment_feature_importance.png")
    print("Total done in %0.3fs" % (time() - t0))


if __name__ == "__main__":
    main()
