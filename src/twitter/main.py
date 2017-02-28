import pandas as pd

from time import time
from sklearn.model_selection import train_test_split, cross_val_score

from tools import plot_significant_features

from model import build_simple_model as build_model


def main():
    df = pd.read_csv('Tweets.csv')

    X = df['text'].values

    y = df['airline_sentiment']
    y = y.map({'negative': -1, 'neutral': 0, 'positive': 1}).values

    pipeline = build_model()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    t0 = time()

    pipeline.fit(X_train, y_train)
    print("Training done in %0.3fs" % (time() - t0))
    print()

    scores = cross_val_score(pipeline, X, y)
    score = scores.mean()
    print("Score:{}".format(score))
    plot_significant_features(pipeline=pipeline,
                              file_name="sentiment_feature_importance.png")
    print("Total done in %0.3fs" % (time() - t0))


if __name__ == "__main__":
    main()
