import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, \
    batch_norm, dropout

from agentnet.resolver import ProbabilisticResolver

from agentnet.agent import Agent


# image observation at current tick goes here, shape = (sample_i,x,y,color)
def build_agent(env, state_size):
    observation_layer = InputLayer((None, state_size))

    net = DenseLayer(observation_layer, 100, name='dense1')
    # net = DenseLayer(net, 256, name='dense2')

    # a layer that predicts Qvalues

    policy_layer = DenseLayer(net,
                              num_units=env.action_space.n,
                              nonlinearity=lasagne.nonlinearities.softmax,
                              name="q-evaluator layer")

    V_layer = DenseLayer(net, 1, nonlinearity=None,
                         name="state values")

    # Pick actions at random proportionally to te probabilities
    action_layer = ProbabilisticResolver(policy_layer,
                                         name="e-greedy action picker",
                                         assume_normalized=True)

    # all together
    agent = Agent(observation_layers=observation_layer,
                  policy_estimators=(policy_layer, V_layer),
                  action_layers=action_layer)

    return agent, action_layer, V_layer
