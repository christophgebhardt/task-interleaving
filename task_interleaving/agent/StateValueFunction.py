class StateValueFunction:
    """
    Abstract state value function
    """

    def __init__(self, num_actions):
        """
        Constructor
        :param num_actions: number of actions in state value function
        """
        self.num_actions = num_actions

    def update(self, state, action, value):
        """
        Updating the respective state action value
        :param state: state
        :param action: action
        :param value: new value
        :return:
        """
        raise NotImplementedError("This is an abstract state-value-function class.")

    def get_value(self, state, action):
        """
        Returns value of passed state and action
        :param state: state
        :param action: action
        :return: value
        """
        raise NotImplementedError("This is an abstract state-value-function class.")

    def get_available_actions(self, state):
        """
        Returns available actions for passed state
        :param state: state
        :return: array of available actions
        """
        raise NotImplementedError("This is an abstract state-value-function class.")

    def reset(self):
        """
        Resets the state action value function to its initial state
        :return:
        """
        raise NotImplementedError("This is an abstract state-value-function class.")