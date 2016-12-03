""" The ChainWorld enviornment used for RL experiments."""


class ChainWorld(object):
    """The environment called ChainWorld.

    The environment consists of a chain of states. There are 1000 states,
    numbered 1 to 1000 from left to right. All episodes begin near the
    center: state 500. There are two terminal states: one on the far left
    (state 0) and one on the far right (state 1000). State transitions can
    occur from the current state to one of the 100 adjacent states to the
    right, or to one of the 100 adjacent states to the left, all with equal
    probability. If there are fewer than 100 states on either side, then the
    remaining probability mass is assigned to the transition to the terminal
    state on that side.

    Attributes:
        ACTIONS (list): List of available actions. An action is an integer
            which represents the index of the state to which this action
            would cause a transition to.
        num_states (int): Number of states in the ChainWorld
        start (int): Index of the start state.
        state (int): Current state of the agent.
        transition_width (int): The maximum number of states on each side of
            the current state that the agent can transition to.
    """
    ACTIONS = []

    def __init__(self, num_states, transition_width):
        """
        Args:
            num_states (int): Number of states in the ChainWorld.
            transition_width (int): The maximum number of states on each side
                of the current state that the agent can transition to.
        """
        self.num_states = num_states
        self.transition_width = transition_width
        self.start = int(num_states / 2)
        self.state = self.start

        self.ACTIONS = range(0, num_states)

    @property
    def leftmost(self):
        """Index of left-most state -- a terminal state.

        Returns:
            int: Index of left-most state.
        """
        return 0

    @property
    def rightmost(self):
        """Index of right-most state -- a terminal state.

        Returns:
            int: Index of right-most state.
        """
        return self.num_states - 1

    def state_iterator(self):
        """Iterate over the states from left (state 0) to right (state 1).

        Yields:
            int: Index of state.
        """
        for state in xrange(self.num_states):
            yield state

    def reset(self):
        """Resets the environment by setting the current state to the start
        state.
        """
        self.state = self.start

    def take_action_and_get_reward(self, action):
        """Executes the give action in the environment by changing the state
        of the environment. The reward associated with the transition is
        returned.

        Args:
            action (int): Action to be executed.

        Returns:
            float: Reward associated with the transition.

        Raises:
            ValueError: If the action requested is illegal.
        """
        if action not in self.ACTIONS:
            raise ValueError("Illegal Action: " + str(action))
        else:
            prev_state = self.state
            self.state = action
            return self.reward(prev_state, self.state)

    def reward(self, current_state, next_state):
        """Returns the reward associated with the transition from a given
        state to a next state.

        Args:
            current_state (int): State from which the transition happens.
            next_state (int): State to which the transition happens.

        Returns:
            float: Reward associated with the transition.
        """
        if (next_state == self.rightmost and
                not self.is_terminal(current_state)):
            return 1.
        elif (next_state == self.leftmost and
                not self.is_terminal(current_state)):
            return -1.
        else:
            return 0.

    def is_terminal(self, state):
        if state == self.leftmost or state == self.rightmost:
            return True
        else:
            return False
