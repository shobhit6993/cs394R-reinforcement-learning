""" The TD(\lambda) algorithm for value-estimation using linear function
approximation for states."""

import numpy as np

import utils


class TD(object):
    """Implements the TD(\lambda) algorithm for value-estimation.

    The algorithm and the terminology of Section 12.2 on "TD(\lambda)" in
    the 2nd edition of "Reinforcement Learning: An Introduction" by Sutton and
    Barto is used.

    A linear function approximation is used for value-function.

    Attributes:
        msve (list of float): List of Mean-Square Value Error of the algorithm
            after each episode.
        theta (numpy.ndarray): The estimated value of parameters of the
            value-function.
    """

    def __init__(self, policy, features, true_values, alpha, gamma, lambda_):
        """
        Args:
            policy (:obj:`Policy`): The policy whose value-function needs to
                be estimated.
            features (:obj:`StateFeatures`): Feature function for states.
            true_values (numpy.ndarray): The true state-values calculated
                externally. These are used only for MSVE calculations,
                and are not provided to LSTD.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            lambda_ (TYPE): Description
        """
        self._policy = policy
        # The RL environment; in this case, the ChainWorld
        self._env = policy.env
        self._features = features
        # The eligibility trace vector.
        self._e = np.zeros(self._features.dimensions)
        self._alpha = alpha
        self._gamma = gamma
        self._lambd = lambda_
        self._true_values = true_values
        self.theta = self._initialize_theta()
        self.msve = []

    def run(self, num_episodes):
        """Runs the TD(\lambda) algorithm for a specified number of episodes.

        Args:
            num_episodes (int): Number of episodes to run.
        """
        for _ in xrange(num_episodes):
            self._env.reset()
            curr_state = self._env.state
            while not self._env.is_terminal(curr_state):
                reward = self._policy.take_action_and_get_reward()
                next_state = self._env.state
                self._update_theta(curr_state, reward, next_state)
                curr_state = next_state

            self._calc_msve()
            self._alpha *= 0.99

    def _calc_msve(self):
        """Calculates the MSVE between the true state-values and the current
        value-estimates The calculates MSVE is added to the `msve` list.
        """
        v = []
        for state in self._env.state_iterator():
            feature_vector = self._features.vector(state)
            v.append(utils.state_value(feature_vector, self.theta))

        self.msve.append(utils.rmse(v, self._true_values))

    def _update_theta(self, curr_state, reward, next_state):
        """Performs one step of paramter updates.

        Args:
            curr_state (int): Current state of the agent.
            reward (float): Reward experienced by the current transition.
            next_state (int): State the agent lands in following the
                transition.
        """
        curr_state_fv = self._features.vector(curr_state)
        next_state_fv = self._features.vector(next_state)
        curr_state_value = utils.state_value(curr_state_fv, self.theta)
        next_state_value = utils.state_value(next_state_fv, self.theta)

        td_error = reward + self._gamma * next_state_value - curr_state_value
        gradient = self._features.vector(curr_state)
        self._e = self._gamma * self._lambd * self._e + gradient

        self.theta += self._alpha * td_error * self._e

    def _initialize_theta(self):
        """Returns an initial value for `theta` parameter.

        Returns:
            numpy.ndarray: Initial value for `theta`.
        """
        return np.zeros(self._features.dimensions)
