from inspect import getsourcelines


class SmartPipeline:
    def __init__(self):
        self.units = []
        self.size = 0

    def __add__(self, other):
        if isinstance(other, tuple):
            self.units.append(other)
        else:
            self.units.append((str(self.size), other))
        self.size += 1

    def fit_transform(self, X):
        for unit in self.units:
            source_hash = getsourcelines(unit.__class__)
            X = unit.fit_transform(X)
        return X
