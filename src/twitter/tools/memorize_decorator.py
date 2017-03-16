from sklearn.base import BaseEstimator, TransformerMixin

from inspect import getsourcelines
import hashlib
from os.path import exists
import pickle


class MemorizeDecorator(BaseEstimator, TransformerMixin):
    def __init__(self, algorithm, fit=True, transform=True):
        self.transformer = algorithm
        self.do_smart_fit = fit
        self.do_smart_transform = transform

    def get_dump_filename(self, X, params, **kwargs):
        source_code = str(getsourcelines(self.transformer.__class__))
        dict_to_hash = {
            'algo_src': source_code,
            'X': X,
            'algo_params': params,
            'kwargs': kwargs,
        }
        str_to_hash = str(dict_to_hash).encode('utf-8')
        hash_value = hashlib.md5(str_to_hash).hexdigest()
        return "dumps/" + hash_value + ".pkl"

    def dump(self, Xt, filename):
        with open(filename, "wb") as file:
            pickle.dump(Xt, file)

    def load(self, filename):
        if not exists(filename):
            raise Exception("No file %s" % filename)

        with open(filename, "rb") as file:
            data = pickle.load(file)
        return data

    def smart_fit(self, X, args=None):
        if args is None:
            args = dict()
        filename = self.get_dump_filename(X, args, memo_type='algo')
        if exists(filename):
            self.transformer = self.load(filename)
        else:
            self.transformer.fit(X, **args)
            self.dump(self.transformer, filename)
        return self

    def smart_transform(self, X, args=None):
        if args is None:
            args = dict()
        filename = self.get_dump_filename(X, args, memo_type='data')
        if exists(filename):
            return self.load(filename)
        new_Xt = self.transformer.transform(X, **args)
        self.dump(new_Xt, filename)
        return new_Xt

    # Overloading default methods

    def transform(self, X, **kwargs):
        if self.do_smart_transform:
            return self.smart_transform(X, kwargs)
        return self.transformer.transform(X, **kwargs)

    def fit_transform(self, X, **kwargs):
        return self.fit(X, **kwargs).transform(X, **kwargs)

    def fit(self, X, **kwargs):
        if self.do_smart_fit:
            return self.smart_fit(X, kwargs)
        return self.transformer.fit(X, **kwargs)

    # Running other methods as in decorated
    def __getattr__(self, method_name):
        return getattr(self.transformer, method_name)
