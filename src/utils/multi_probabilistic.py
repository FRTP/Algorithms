import theano.tensor as T
import theano.tensor.shared_randomstreams as random_streams

from agentnet.resolver import BaseResolver


class MultiProbabilisticResolver(BaseResolver):
    """
    Behavior is similar to agentnet.resolver.ProbabilisticResolver
    but allows multidimensional output

    """

    def __init__(self, incoming, assume_normalized=False, seed=1234,
                 output_dtype='int32',
                 name='MultiProbabilisticResolver'):

        self.assume_normalized = assume_normalized

        self.rng = random_streams.RandomStreams(seed)

        super(MultiProbabilisticResolver, self).__init__(
            incoming,
            name=name,
            output_dtype=output_dtype)

    def get_output_for(self, policy, greedy=False, **kwargs):
        if greedy:
            # greedy branch
            chosen_action_ids = T.argmax(policy, axis=-1).astype(
                self.output_dtype)

        else:

            if self.assume_normalized:
                probas = policy
            else:
                probas = policy / T.sum(
                    policy,
                    axis=-1,
                    keepdims=True)

            # p1, p1+p2, p1+p2+p3, ... 1
            cum_probas = T.cumsum(probas, axis=-1)

            rnd_shape = T.stack([*policy.shape[:-1], 1])

            batch_randomness = self.rng.uniform(
                low=0.,
                high=1.,
                size=rnd_shape)
            batch_randomness = T.repeat(
                batch_randomness,
                policy.shape[-1] - 1,
                axis=-1)

            chosen_action_ids = T.sum(
                (batch_randomness > cum_probas[:, :, :-1]),
                axis=-1,
                dtype=self.output_dtype)

        return chosen_action_ids

    def get_output_shape_for(self, input_shape):
        """returns shape of layer output"""
        return input_shape[:-1]
