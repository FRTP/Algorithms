import re

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing import common_regex as cre
from preprocessing.dictionary import emo


class Replacer(BaseEstimator, TransformerMixin):
    def __init__(self):
        smile_list = emo['SMILE'] + emo['LAUGH'] + emo['WINK']
        sadface_list = emo['FROWN'] + emo['CRY']
        heart_list = emo['LOVE']

        self.patterns = [
            # should be done before others
            (cre.re_allcaps, cre.allcaps_repl),
            (cre.re_from_list(heart_list), cre.heart_repl),
            (cre.re_from_list(smile_list), cre.smile_repl),
            (cre.re_from_list(sadface_list), cre.heart_repl),
            (cre.re_money, cre.money_repl),
            (cre.re_number, cre.number_repl),
            (cre.re_url, cre.url_repl),
            (cre.re_hashtag, cre.hashtag_repl),
            (cre.re_user, cre.user_repl),
            (cre.re_repeat, cre.repeat_repl),
            (cre.re_sign_repeat, cre.sign_repeat_repl),
        ]

    def single_transform(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s

    def transform(self, X):
        X_vectors = []
        for x in X:
            x_new = self.single_transform(x)
            X_vectors.append(x_new)

        return np.array(X_vectors)

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X).transform(X)

    def fit(self, X, y=None):
        # self.transform(X)
        return self
