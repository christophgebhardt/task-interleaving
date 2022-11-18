class Environment:
    """
    Abstract environment class defining its functionality
    """

    def __init__(self):
        self.test = ""

    def perform_action(self, action):
        """
        Executes an action in the respective environment
        :param action: action to execute
        :return:
        """
        raise NotImplementedError("This is an abstract environment.")

    def get_observation(self):
        """
        Returns the currently available observation of the environment
        :return: observation
        """
        raise NotImplementedError("This is an abstract environment.")

    def reset(self):
        """
        Resets the environment to its start state
        :return:
        """
        raise NotImplementedError("This is an abstract environment.")
