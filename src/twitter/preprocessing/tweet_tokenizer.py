import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

from sklearn.base import BaseEstimator, TransformerMixin


class TweetTokenizerTransformer(BaseEstimator, TransformerMixin, TweetTokenizer):
    def transform(self, X):
        X_vectors = []
        for x in X:
            tokens = super().tokenize(x)
            X_vectors.append(tokens)

        return np.array(X_vectors)

    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X)

    def fit(self, X, y=None):
        self.transform(X)
        return self
