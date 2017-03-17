import numpy as np
import nltk
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin


class Tokenizer(BaseEstimator, TransformerMixin):
    def tokenize(self, text):
        tokens = []
        for sent in nltk.sent_tokenize(text, language='english'):
            for word in nltk.word_tokenize(sent, language='english'):
                if len(word) < 2:
                    continue
                if word in stopwords.words('english'):
                    continue
                tokens.append(word)
        return tokens

    def transform(self, X):
        X_vectors = []
        for x in X:
            tokens = self.tokenize(x)
            X_vectors.append(tokens)

        return np.array(X_vectors)

    def fit(self, X, y=None):
        # self.transform(X)
        return self
