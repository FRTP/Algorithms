import re
import string

from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    _numeric = re.compile("(\$)?\d+([\.,]\d+)*")
    _links = re.compile(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        "[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    _trans_table = str.maketrans({key: None for key in string.punctuation})

    def clean_text(self, tokens):
        for i, w in enumerate(tokens):
            if TextCleaner._numeric.match(w):
                tokens[i] = w.translate(TextCleaner._trans_table)
            elif TextCleaner._links.match(w):
                tokens[i] = '_LINK_'
            else:
                continue
        return tokens

    def transform(self, X):
        for i, item in enumerate(X):
            tokenized = item.split()
            X[i] = " ".join(self.clean_text(tokenized))

        return X

    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X)

    def fit(self, X, y=None):
        self.transform(X)
        return self
