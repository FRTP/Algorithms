import string

from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing.common_regex import re_number, re_url


class TextCleaner(BaseEstimator, TransformerMixin):

    def __init__(self):
        nones = [None for _ in string.punctuation]
        d = dict(zip(string.punctuation, nones))
        self.trans_table = string.punctuation.maketrans(d)

    def clean_text(self, tokens):
        for i, w in enumerate(tokens):
            if re_number.match(w):
                tokens[i] = w.translate(self.trans_table)
            elif re_url.match(w):
                tokens[i] = '_LINK_'
            else:
                continue
        return tokens

    def transform(self, X):
        for i, item in enumerate(X):
            tokens = item.split()
            X[i] = " ".join(self.clean_text(tokens))

        return X

    def fit(self, X, y=None):
        # self.transform(X)
        return self
