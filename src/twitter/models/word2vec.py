import logging

import numpy as np
from gensim.models import Word2Vec
from gensim.matutils import unitvec

from sklearn.base import BaseEstimator, TransformerMixin


def load_model(filename):
        wv = Word2Vec.load_word2vec_format(filename)
        wv.init_sims(replace=True)
        return wv


class Word2VecAvarager(BaseEstimator, TransformerMixin):
    def __init__(self, filename=None, wv=None):
        self.filename = filename
        self.wv = wv

    def load(self):
        if self.wv is None:
            if self.filename is None:
                raise Exception("missing wv and filename field")
            self.wv = Word2Vec.load_word2vec_format(self.filename)
            self.wv.init_sims(replace=True)

    def word_averaging(self, words):
        vecs = []

        for word in words:
            if isinstance(word, np.ndarray):
                vecs.append(word)
            elif word in self.wv.wv.vocab:
                id = self.wv.wv.vocab[word].index
                vecs.append(self.wv.wv.syn0norm[id])

        if not vecs:
            logging.warning("cannot compute similarity : %s", words)
            # FIXME: remove these examples in pre-processing
            return np.zeros(self.wv.layer1_size, )

        vec = np.array(vecs).sum(axis=0)
        vec = unitvec(vec).astype(np.float32)
        return vec

    def transform(self, X):
        self.load()
        X_vectors = []
        for tokens in X:
            X_vectors.append(self.word_averaging(tokens))

        return np.array(X_vectors)

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X).transform(X)

    def fit(self, X, y=None):
        # self.transform(X)
        return self
