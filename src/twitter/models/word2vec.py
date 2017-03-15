import logging

import numpy as np
from gensim.models import Word2Vec
from gensim.matutils import unitvec

from sklearn.base import BaseEstimator, TransformerMixin


class Word2VecAvarager(BaseEstimator, TransformerMixin):
    def __init__(self, filename):
        logging.info("Loading Word2Vec ...")
        self.wv = Word2Vec.load_word2vec_format(filename)
        self.wv.init_sims(replace=True)

    def word_averaging(self, words):
        mean = []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in self.wv.wv.vocab:
                mean.append(self.wv.wv.syn0norm[self.wv.wv.vocab[word].index])

        if not mean:
            logging.warning("cannot compute similarity with no input %s", words)
            # FIXME: remove these examples in pre-processing
            return np.zeros(self.wv.layer1_size, )

        mean = unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def transform(self, X):
        X_vectors = []
        for tokens in X:
            X_vectors.append(self.word_averaging(tokens))

        return np.array(X_vectors)

    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X)

    def fit(self, X, y=None):
        self.transform(X)
        return self
