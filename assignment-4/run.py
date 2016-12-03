import pylab

from chainworld import ChainWorld
from policy import Policy
from state_features import StateFeatures
from td import TD
from lstd import LSTD
import utils


def estimate_values_by_td(env, policy, features, true_values, lambda_, alpha):
    """Computes value-estimates using TD(\lambda) algorithm.

    Args:
        env (ChainWorld): The RL environment.
        policy (Policy): The policy that needs to be evaluated.
        features (StateFeatures): Feature function for states of the env.
        true_values (numpy.ndarray): The true state-values for given policy.
        lambda_ (float): \lambda parameter of TD(\lambda).
        alpha (float): Learning rate.

    Returns:
        list, list: Final value estimates of all states and MSVE errors of the
            algorithm's estimates over the course of it's execution.
    """
    td = TD(policy=policy, features=features, gamma=0.95,
            alpha=0.1, lambda_=lambda_, true_values=true_values)
    td.run(num_episodes)

    v = []
    for state in env.state_iterator():
        feature_vector = features.vector(state)
        v.append(utils.state_value(feature_vector, td.theta))
    return v, td.msve


def estimate_values_by_lstd(env, policy, features, true_values, epsilon):
    """Computes value-estimates using LSTD algorithm.

    Args:
        env (ChainWorld): The RL environment.
        policy (Policy): The policy that needs to be evaluated.
        features (StateFeatures): Feature function for states of the env.
        true_values (numpy.ndarray): The true state-values for given policy.
        lambda_ (float): \epsilon parameter of LSTD.

    Returns:
        list, list: Final value estimates of all states and MSVE errors of the
            algorithm's estimates over the course of it's execution.
    """
    lstd = LSTD(policy=policy, features=features, gamma=0.95,
                true_values=true_values, epsilon=epsilon)
    lstd.run(num_episodes)

    v = []
    for state in env.state_iterator():
        feature_vector = features.vector(state)
        v.append(utils.state_value(feature_vector, lstd.theta))
    return v, lstd.msve


num_states = 1000
cluster_size = 100
num_episodes = 1000
env = ChainWorld(num_states, cluster_size)
policy = Policy(env)
features = StateFeatures(env, cluster_size)

true_values = policy.find_optimal_v()

td_0_values, msve_td_0 = estimate_values_by_td(
    env, policy, features, true_values, 0., 0.07)
td_lambda_values, msve_td_lambda = estimate_values_by_td(
    env, policy, features, true_values, 0.5, 0.01)
lstd_values, msve_lstd = estimate_values_by_lstd(
    env, policy, features, true_values, 0.09)

states = [s for s in env.state_iterator()]
pylab.plot(states, true_values, label='true_v')
pylab.plot(states, td_0_values, label="td(0)")
pylab.plot(states, td_lambda_values, label="td(0.5)")
pylab.plot(states, lstd_values, label="lstd-0.09")
pylab.xlim(0, 1001)
pylab.xlabel("State")
pylab.ylabel("Value of state")
pylab.legend(loc='best')
pylab.show()

pylab.plot(range(0, num_episodes), msve_td_0, label='td(0)')
pylab.plot(range(0, num_episodes), msve_td_lambda, label='td(0.5)')
pylab.plot(range(0, num_episodes), msve_lstd, label='lstd-0.09')
pylab.legend(loc='best')
pylab.ylim(0, 1.5)
pylab.xlabel("Number of episodes")
pylab.ylabel("MSVE")
pylab.show()
