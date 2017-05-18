import numpy as np

import lasagne
from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer, \
    NonlinearityLayer, batch_norm, dropout

# from agentnet.resolver import ProbabilisticResolver

from utils.multi_probabilistic import MultiProbabilisticResolver

from agentnet.agent import Agent


# image observation at current tick goes here, shape = (sample_i,x,y,color)
def build_agent(action_shape, state_shape):
    observation_layer = InputLayer((None, *state_shape))

    net = DenseLayer(observation_layer, 100, name='dense1')
    # net = DenseLayer(net, 256, name='dense2')

    # a layer that predicts Qvalues

    policy_layer_flattened = DenseLayer(net,
                                        num_units=np.prod(action_shape),
                                        nonlinearity=lasagne.nonlinearities.softmax,
                                        name="q-evaluator layer")

    policy_layer = ReshapeLayer(policy_layer_flattened,
                                ([0], *action_shape))

    V_layer = DenseLayer(net, 1, nonlinearity=None,
                         name="state values")

    # Pick actions at random proportionally to te probabilities
    action_layer = MultiProbabilisticResolver(policy_layer,
                                              name="e-greedy action picker",
                                              assume_normalized=True)

    # all together
    agent = Agent(observation_layers=observation_layer,
                  policy_estimators=(policy_layer_flattened, V_layer),
                  action_layers=action_layer)

    return agent, action_layer, V_layer
