import numpy as np
from nltk.tokenize import TweetTokenizer

from sklearn.base import BaseEstimator, TransformerMixin


class TweetTokenizerTransformer(BaseEstimator,
                                TransformerMixin,
                                TweetTokenizer):
    def __init__(self, lower=True):
        super().__init__()
        self.lower = lower

    def single_transform(self, text):
        if self.lower:
            text = text.lower()
        tokens = super().tokenize(text)
        return tokens

    def transform(self, X):
        X_vectors = []
        for x in X:
            tokens = self.single_transform(x)
            X_vectors.append(tokens)

        return np.array(X_vectors)

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X).transform(X)

    def fit(self, X, y=None):
        # self.transform(X)
        return self
