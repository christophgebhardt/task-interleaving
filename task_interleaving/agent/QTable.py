import numpy as np

from task_interleaving.agent.StateValueFunction import StateValueFunction


class QTable(StateValueFunction):
    """
    QTable to represent state value function
    """

    def __init__(self, num_actions, num_states):
        """
        Constructor
        :param num_actions: number of actions
        :param num_states: number of states
        """
        StateValueFunction.__init__(self, num_actions)
        self.num_states = num_states
        self.q = np.zeros((num_states, num_actions))
        # self.q = np.random.rand(num_states, num_actions)
        self.alpha = 0.1

    def update(self, state, action, value):
        """
        Updates the q table for a given state and action towards the target value.
        :param state: index of state
        :param action: index of action
        :param value: target value
        :return:
        """
        self.q[state[0], action] += self.alpha * (value - self.q[state[0], action])

    def get_value(self, state, action):
        return self.q[state[0], action]

    def get_available_actions(self, state):
        return np.arange(self.num_actions + 1)

    def reset(self):
        self.q = np.zeros((self.num_states, self.num_actions))
