# a simplified version of
# https://gist.github.com/ferrine/a003ace716c278ab87669f2fbd37727b
import numpy as np
from functools import wraps

from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne

__all__ = ['NormalApproximation', 'bbpwrap', 'sample_output']


class NormalApproximation(object):
    def __init__(self, mu_init=lasagne.init.Normal(1),
                 rho_init=lasagne.init.Constant(0)):
        self.mu_init = mu_init
        self.rho_init = rho_init
        self.rng = RandomStreams(np.random.randint(0, 2147462579))

    def __call__(self, layer, spec, shape, **tags):
        # case when user uses default init specs
        if not isinstance(spec, dict):
            spec = {'mu': spec}
        # important!
        # we declare that params we add next
        # are the ones we need to fit the distribution
        tags['variational'] = True

        rho_spec = spec.get('rho', self.mu_init)
        mu_spec = spec.get('mu', self.rho_init)

        rho = layer.add_param(rho_spec, shape, name='bnn.rho',
                              rho=True, **tags)
        mean = layer.add_param(mu_spec, shape, name='bnn.mu', mu=True,
                               **tags)

        W = mean + T.log1p(T.exp(rho)) * self.rng.normal(shape, std=1)

        return W


def bbpwrap(approximation=NormalApproximation()):
    def decorator(cls):
        def add_param_wrap(add_param):
            @wraps(add_param)
            def wrapped(self, spec, shape, name=None, **tags):
                # we should take care about some user specification
                # to avoid bbp hook set tags['variational'] = True
                if not tags.get('trainable', True) or tags.get(
                        'variational', False):
                    return add_param(self, spec, shape, name, **tags)
                else:
                    # they don't need to be regularized, strictly
                    tags['regularizable'] = False
                    param = self.approximation(self, spec, shape,
                                               **tags)
                    return param

            return wrapped

        cls.approximation = approximation
        cls.add_param = add_param_wrap(cls.add_param)
        return cls

    return decorator


def sample_output(output_layer, input_dict,
                  n_samples=10,
                  aggregate=lambda v: T.mean(v, axis=1),
                  **kwargs):
    """get samples from neural network and aggregate over them"""
    repeated_input_dict = {
        layer: T.repeat(inp, n_samples, axis=0)
        for layer, inp in input_dict.items()
        }

    out_ravel = lasagne.layers.get_output(output_layer,
                                          repeated_input_dict,
                                          **kwargs)

    return aggregate(out_ravel.reshape(
        (out_ravel.shape[0] // n_samples, n_samples,) + tuple(
            out_ravel.shape[1:])))
