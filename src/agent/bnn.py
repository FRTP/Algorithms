import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, EmbeddingLayer

from agent.bnn_utils import bbpwrap, NormalApproximation
from agent.curiosity import compile_vime_reward


@bbpwrap(NormalApproximation())
class BayesDenseLayer(DenseLayer):
    pass


@bbpwrap(NormalApproximation())
class BayesEmbLayer(EmbeddingLayer):
    pass


class BNN:
    def __init__(self, state_shape, action_shape, action_emb_shape,
                 replay):
        self.curiosity = 0.01
        target_rho = 1

        l_state = InputLayer((None, *state_shape),
                             name='state var')
        l_action = InputLayer((None, *action_shape),
                              input_var=T.imatrix(),
                              name='actions var')

        # TODO: HUGE FIX NEEDED HERE
        l_action_emb = BayesEmbLayer(l_action, *action_emb_shape)

        l_action_emb_flat = lasagne.layers.flatten(l_action_emb)

        # l_action_emb_flat = BayesEmbLayer(
        #     l_action,
        #     input_size=action_shape[0],
        #     output_size=np.prod(action_emb_shape))

        l_concat = lasagne.layers.concat([l_action_emb_flat, l_state])

        print("A_emb: ", l_action_emb.output_shape)
        print("S: ", l_state.output_shape)
        print("A_emb_flat: ", l_action_emb_flat.output_shape)
        print("Concat: ", l_concat.output_shape)

        l_dense = BayesDenseLayer(
            l_concat,
            num_units=50,
            nonlinearity=lasagne.nonlinearities.tanh,
            name='dense 1')

        l_out = BayesDenseLayer(
            l_dense,
            num_units=state_shape[0],
            nonlinearity=None)

        params = lasagne.layers.get_all_params(l_out, trainable=True)

        # training
        pred_states = lasagne.layers.get_output(l_out)
        next_states = T.matrix("next states")
        mse = lasagne.objectives.squared_error(pred_states,
                                               next_states).mean()

        # logposterior with simple regularization on rho cuz we R lazy
        reg = sum(
            [lasagne.objectives.squared_error(rho, target_rho).mean()
             for rho in
             lasagne.layers.get_all_params(l_out, rho=True)])

        loss = mse + 0.01 * reg

        updates = lasagne.updates.adam(loss, params)

        self.train_step = theano.function(
            [l_state.input_var, l_action.input_var, next_states],
            loss, updates=updates)

        # sample random sessions from pool
        observations, = replay.observations
        actions, = replay.actions
        observations_flat = observations[:, :-1].reshape(
            (-1,) + tuple(observations.shape[2:]))
        actions_flat = actions[:, :-1].reshape((-1,))
        next_observations_flat = observations[:, 1:].reshape(
            (-1,) + tuple(observations.shape[2:]))
        self.sample_from_pool = theano.function(
            [],
            [observations_flat,
             actions_flat,
             next_observations_flat])

        # curiosity reward### aka KL(qnew,qold)
        self.get_vime_reward_elwise = compile_vime_reward(
            l_out,
            l_state,
            l_action,
            params,
            n_samples=10)

        self.vime_reward_ma = 10.

    def add_vime_reward(self, observations, actions, rewards,
                        is_alive, h0=0):
        assert isinstance(observations, np.ndarray)
        observations_flat = observations[:, :-1].reshape(
            (-1,) + observations.shape[2:]).astype('float32')
        actions_flat = actions[:, :-1].reshape((-1,)).astype('int32')
        next_observations_flat = observations[:, 1:].reshape(
            (-1,) + observations.shape[2:]).astype('float32')

        vime_rewards = self.get_vime_reward_elwise(
            observations_flat,
            actions_flat,
            next_observations_flat)
        vime_rewards = np.concatenate(
            [vime_rewards.reshape(rewards[:, :-1].shape),
             np.zeros_like(rewards[:, -1:]), ], axis=1)
        # normalize by moving average
        self.vime_reward_ma = \
            0.99 * self.vime_reward_ma + 0.01 * vime_rewards.mean()

        surrogate_rewards = rewards + \
            self.curiosity / self.vime_reward_ma * vime_rewards
        return (
            observations, actions, surrogate_rewards, is_alive, h0)
