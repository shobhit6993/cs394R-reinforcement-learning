"""Feature function for states of ChainWorld."""

from math import ceil
import numpy as np


class StateFeatures(object):
    """Feature function for ChainWorld states.

    State-aggregation scheme, the simplest generalizating function
    approximation, is used to define features for states. States are clustered
    into groups. All states in a group share a common feature vector.

    Attributes:
        cluster_size (int): The size of each state-group for state-aggregation.
        dimensions (int): The dimension of feature vectors for states.
    """

    def __init__(self, env, cluster_size):
        """
        Args:
            env (ChainWorld): The enviornment.
            cluster_size (TYPE): The size of each state-group for
                state-aggregation.
        """
        self.cluster_size = cluster_size
        self.dimensions = int(
            ceil(float(env.num_states) / self.cluster_size))
        self._env = env
        # The mapping defining the feature function.
        self._mapping = {}

        self._build_features()

    def vector(self, state):
        """Returns the feature vector for the requested state.

        Args:
            state (int): The state whose feature vector is required.

        Returns:
            numpy.ndarray: State's feature vector.
        """
        try:
            return self._mapping[state]
        except KeyError:
            print("Invalid state: " + str(state))
            raise

    def _build_features(self):
        """Builds the feature function for the states of the `_env`
        environment using state-aggregation scheme.
        """
        vector = None
        for i, state in enumerate(self._env.state_iterator()):
            if state == self._env.leftmost or state == self._env.rightmost:
                self._mapping[state] = np.zeros(self.dimensions)
            if i % self.cluster_size == 0:
                vector = np.zeros(self.dimensions)
                vector[i / self.cluster_size] = 1.
            self._mapping[state] = vector
