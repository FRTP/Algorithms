from sklearn.base import BaseEstimator, TransformerMixin, _pprint
import logging

from inspect import getsourcelines
import hashlib
import os
import pickle


def dump(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load(filename):
    if not os.path.exists(filename):
        raise Exception("No file %s" % filename)

    with open(filename, "rb") as file:
        data = pickle.load(file)
    return data


class MemoDecorator(BaseEstimator, TransformerMixin):
    DUMP_PATH = 'dumps/'

    def __init__(self, algorithm,
                 memorize_fit=True,
                 memorize_transform=True,
                 **kwargs):
        self.algorithm = algorithm
        self.memorize_fit = memorize_fit
        self.memorize_transform = memorize_transform
        self.set_params(**kwargs)

    def get_dump_path(self, X, params, **kwargs):
        source_lines = getsourcelines(self.algorithm.__class__)
        dict_to_hash = {
            'algo_src_lines': source_lines,
            'algo_args': params,
            'X': X,
            'kwargs': kwargs,
        }
        str_to_hash = str(dict_to_hash).encode('utf-8')
        hash_value = hashlib.md5(str_to_hash).hexdigest()
        return MemoDecorator.DUMP_PATH + hash_value + ".pkl"

    def smart_method(self, X, args, method_name):
        if args is None:
            args = dict()

        dump_path = self.get_dump_path(X, args, method=method_name)
        if os.path.exists(dump_path):
            logging.getLogger(self.class_name()).info(
                "loading results for %s()" % method_name)
            returned = load(dump_path)
        else:
            method = getattr(self.algorithm, method_name)
            returned = method(X, **args)
            logging.getLogger(self.class_name()).info(
                "dumping results for %s()" % method_name)
            dump(returned, dump_path)
        return returned

    # Overloading default TransformerMixin methods

    def transform(self, X, **kwargs):
        if self.memorize_transform:
            returned = self.smart_method(X, kwargs, 'transform')
        else:
            returned = self.algorithm.transform(X, **kwargs)
        return returned

    def predict(self, X, **kwargs):
        if self.memorize_transform:
            return self.smart_method(X, kwargs, 'predict')
        return self.algorithm.predict(X, **kwargs)

    def fit(self, X, y=None, **kwargs):
        if self.memorize_fit:
            kwargs['y'] = y
            self.algorithm = self.smart_method(X, kwargs, 'fit')
        else:
            self.algorithm.fit(X, y, **kwargs)
        return self

    # overloading BaseEstimators methods

    def get_real_params(self):
        attrs = [attr for attr in dir(self)
                 if not callable(getattr(self, attr)) and
                 not attr.upper() == attr and
                 not attr.startswith("__")]
        return dict([(attr, getattr(self, attr)) for attr in attrs])

    def set_real_params(self, **params):
        for k in params:
            setattr(self, k, params[k])

    def get_params(self, deep=True):
        out = self.get_real_params()
        out.update(self.algorithm.get_params(deep))
        return out

    def set_params(self, **params):
        memo_keys = list(self.get_real_params().keys())
        memo_params, trans_params = dict_split(params, memo_keys)
        self.set_real_params(**memo_params)
        self.algorithm.set_params(**trans_params)

    # overloading other methods

    def __getattr__(self, method_name):
        return getattr(self.algorithm, method_name)

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name,
                           _pprint(self.get_params(deep=False),
                                   offset=len(class_name), ),)

    def __str__(self):
        return self.class_name

    @staticmethod
    def clean_memorized():
        path = MemoDecorator.DUMP_PATH
        files = os.listdir(path)
        logging.info("removing files: {}".format(files))
        for file in files:
            file_path = path + file
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(e)

    def subclass_name(self):
        return self.algorithm.__class__.__name__

    def class_name(self):
        return "Memo@" + self.subclass_name()


def dict_split(d, keys):
    dict1 = {}
    dict2 = {}
    for key in d:
        if key in keys:
            dict1[key] = d[key]
        else:
            dict2[key] = d[key]
    return dict1, dict2


def decorate_pipeline(pipeline):
    # from sklearn.pipeline import Pipeline
    steps = [(name, MemoDecorator(algo))
             for name, algo in pipeline.steps]
    pipeline.steps = steps
    return pipeline
