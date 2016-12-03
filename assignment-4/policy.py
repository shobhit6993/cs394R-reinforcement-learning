"""The defined policy for navigation in ChainWorld domain."""

import numpy as np

from utils import normalize_probabilities


class Policy(object):
    """The policy class for the agent to navigate in the ChainWorld domain.

    Attributes:
        env (ChainWorld): The ChainWorld domain.
    """

    def __init__(self, env):
        """
        Args:
            env (ChainWorld): The ChainWorld domain.
        """
        self.env = env

    def take_action_and_get_reward(self):
        """Executes an action according to the defined policy and returns
        the experienced reward.

        Returns:
            float: The reward given by the environment following the action.
        """
        if self.env.is_terminal(self.env.state):
            return 0

        action_probabilities = self._probability_vector_for_state(
            self.env.state)
        action = np.random.choice(
            self.env.ACTIONS, 1, p=action_probabilities)[0]
        reward = self.env.take_action_and_get_reward(action)
        return reward

    def find_optimal_v(self):
        """Computes the true state-values for the defined policy.

        The true state-values can be computed exactly by solving a system
        of linear equations.

        Returns:
            numpy.ndarray: The true state-values.
        """
        # Row vector for the equation corresponding to value of leftmost
        # terminal state.
        left_vector = np.zeros(self.env.num_states)
        left_vector[self.env.leftmost] = 1.

        # a is the matrix in ax=b. It is built by adding rows corresponding to
        # each linear equation one at a time.
        a = left_vector

        # Construct the rest of the matrix, except the equation corresponding
        # to the value of the rightmost state.
        for state in self.env.state_iterator():
            if state == self.env.leftmost or state == self.env.rightmost:
                continue

            vector = self._probability_vector_for_state(state)
            vector[state] = -1
            a = np.vstack((a, vector))

        # Row vector for the equation corresponding to value of rightmost
        # terminal state.
        right_vector = np.zeros(self.env.num_states)
        right_vector[self.env.rightmost] = 1.
        a = np.vstack((a, right_vector))

        # Construct b
        b = np.zeros(self.env.num_states)
        b[self.env.leftmost] = self.env.reward(None, self.env.leftmost)
        b[self.env.rightmost] = self.env.reward(None, self.env.rightmost)

        v = np.linalg.solve(a, b)
        return v

    def _probability_vector_for_state(self, state):
        """Returns the vector of probabilities of transitioning to every state
        from the given state according to the defined policy.

        Args:
            state (int): The state from which transition probabilities are
                needed.

        Returns:
            numpy.ndarray: Vector of probabilities.
        """
        probabilities = np.zeros(self.env.num_states)
        p = 1. / (2. * self.env.transition_width)
        right_sum = 0
        left_sum = 0
        last_right = 0
        last_left = 0
        for i in xrange(1, self.env.transition_width + 1):
            if state + i <= self.env.rightmost:
                probabilities[state + i] = p
                right_sum += p
                last_right = state + i
            if state - i >= self.env.leftmost:
                probabilities[state - i] = p
                left_sum += p
                last_left = state - i

        if right_sum != 0.5:
            probabilities[last_right] += 0.5 - right_sum
        if left_sum != 0.5:
            probabilities[last_left] += 0.5 - left_sum

        normalize_probabilities(probabilities)
        return probabilities
