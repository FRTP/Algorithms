import re

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing.common_regex import *
from preprocessing.dictionary import emo


class Replacer(BaseEstimator, TransformerMixin):
    def __init__(self):
        smile_list = emo['SMILE'] + emo['LAUGH'] + emo['WINK']
        sadface_list = emo['FROWN'] + emo['CRY']
        heart_list = emo['LOVE']

        self.patterns = [
            # should be done before others
            (re_allcaps, allcaps_repl),
            (re_from_list(heart_list), heart_repl),
            (re_from_list(smile_list), smile_repl),
            (re_from_list(sadface_list), heart_repl),
            (re_money, money_repl),
            (re_number, number_repl),
            (re_url, url_repl),
            (re_hashtag, hashtag_repl),
            (re_user, user_repl),
            (re_repeat, repeat_repl),
            (re_sign_repeat, sign_repeat_repl),
        ]

    def single_transform(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s

    def transform(self, X):
        X_vectors = []
        for x in X:
            x_new = self.single_transform(x).lower()
            X_vectors.append(x_new)

        return np.array(X_vectors)

    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X)

    def fit(self, X, y=None):
        self.transform(X)
        return self
