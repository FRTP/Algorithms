import pandas as pd

from time import time
from sklearn.model_selection import cross_val_score, StratifiedKFold

# from tools.tools import plot_significant_features

# from models.baseline_model import build_model
from models.word2vec_model import build_model

import logging


def main():
    logging.basicConfig(level=logging.DEBUG)

    df = pd.read_csv('data/Tweets.csv')
    X = df['text'].values
    y = df['airline_sentiment']
    y = y.map({'negative': -1, 'neutral': 0, 'positive': 1}).values

    model = build_model()

    t0 = time()

    # using determined test_splits for smart dump/load
    cv = StratifiedKFold(n_splits=3, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv)

    score = scores.mean()
    std = scores.std()
    print("Score: %.4f +- %.4f" % (score, std))
    print("Total done in %0.3fs" % (time() - t0))


if __name__ == "__main__":
    main()
