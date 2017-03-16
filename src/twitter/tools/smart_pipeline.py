from inspect import getsourcelines

from sklearn.pipeline import Pipeline
import hashlib
from os.path import exists


class SmartPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.units = []
        self.size = 0

    def __add__(self, other):
        if isinstance(other, tuple):
            self.units.append(other)
        else:
            self.units.append((str(self.size), other))
        self.size += 1

    def fit_transform(self, X, **kwargs):
        for unit in self.units:
            source_code = getsourcelines(unit.__class__).h
            source_hash = hashlib.sha256(source_code)
            if not exists(source_hash + "data"):
                X = unit.fit_transform(X)
            # TODO: else load data
        return X
