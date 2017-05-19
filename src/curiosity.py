"""
computes the curiosity reward batch-wise
"""
import theano
import theano.tensor as T


def extract_params(bnn_weights, pred_loss, delta=0.01):
    rhos = list(
        filter(lambda p: p.name.endswith('bnn.rho'), bnn_weights))
    mus = list(
        filter(lambda p: p.name.endswith('bnn.mu'), bnn_weights))

    grad_mus = T.grad(pred_loss, mus)
    new_mus = [(mu - delta * grad_mu)
               for mu, grad_mu in zip(mus, grad_mus)]

    grad_rhos = T.grad(pred_loss, rhos)
    new_rhos = [(rho - delta * grad_rho)
                for rho, grad_rho in zip(rhos, grad_rhos)]

    for i in range(len(new_mus)):
        new_mus[i].name = 'new_mu'
    for i in range(len(new_rhos)):
        new_rhos[i].name = 'new_rho'

    mus, new_mus, rhos, new_rhos = list(map(
        lambda variables: T.concatenate(
            [var.ravel() for var in variables]),
        [mus, new_mus, rhos, new_rhos]))

    sigmas = T.log1p(T.exp(rhos))
    new_sigmas = T.log1p(T.exp(new_rhos))

    return mus, new_mus, sigmas, new_sigmas


def get_vime_reward(prev_mu, new_mu, prev_sigma, new_sigma):
    """KL for normal approximation"""
    kl = T.sum((new_sigma / prev_sigma) ** 2)
    kl += 2 * T.sum(T.log(prev_sigma))
    kl -= 2 * T.sum(T.log(new_sigma))
    kl += T.sum(((new_mu - prev_mu) / prev_sigma) ** 2)
    kl *= .5
    kl -= .5 * new_sigma.shape[0]
    return kl


def get_r_vime_on_state(pred_weights, pred_loss, delta=0.01):
    """vime reward (single number) on state or states provided"""
    return get_vime_reward(
        *extract_params(pred_weights, pred_loss, delta=delta))


import lasagne
from bnn import sample_output


def compile_vime_reward(l_prediction, l_prev_states, l_actions,
                        weights,
                        get_loss=lambda pred, real: T.mean(
                            (pred - real) ** 2),
                        n_samples=1,
                        delta=0.01, **kwargs):
    """
    compiles a function that predicts vime reward for each state
    in a batch
    """
    prev_states = T.matrix("previous states")
    actions = T.ivector("actions")
    next_states = T.matrix("next states")
    if n_samples == 1:
        get_bnn = lambda state, action: lasagne.layers.get_output(
            l_prediction,
            inputs={l_prev_states: state[None, :],
                    l_actions: action[None]}, **kwargs)
    else:
        get_bnn = lambda state, action: sample_output(
            l_prediction,
            input_dict={
                l_prev_states: state[None, :],
                l_actions: action[None]},
            n_samples=n_samples,
            **kwargs)

    vime_reward_per_state, auto_updates = theano.map(
        lambda s, a, s_next: get_r_vime_on_state(weights,
                                                 get_loss(
                                                     get_bnn(s, a),
                                                     s_next),
                                                 delta),
        sequences=[prev_states, actions, next_states])

    return theano.function([prev_states, actions, next_states],
                           vime_reward_per_state,
                           updates=auto_updates,
                           allow_input_downcast=True)
