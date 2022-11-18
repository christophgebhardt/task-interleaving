class Task:
    """
    Abstract reinforcement learning task instance. Its a wrapper around the RL environment providing more functionality
    than the environment itself. It forwards relevant calls to the environment and augments it with functions that give
    information on if the task the agent tried to solve is finished and the reward it attained from its previous action
    """

    def __init__(self, environment):
        """
        Constructor
        :param environment: instance of underlying environment
        """
        self.environment = environment

    def get_reward(self):
        """
        Returns reward attainable in its current state
        :return: reward value
        """
        raise NotImplementedError("This is an abstract task.")

    def perform_action(self, action):
        """
        Performs the passed action in the underlying environment
        :param action: action index
        :return:
        """
        self.environment.perform_action(action)

    def get_state(self):
        """
        Returns the state the underlying environment is in
        :return: state representation
        """
        raise NotImplementedError("This is an abstract task.")

    def is_finished(self):
        """
        Returns a flag indicating if the task is finished or not
        :return:
        """
        raise NotImplementedError("This is an abstract task.")

    def reset(self):
        """
        Resets the task to its initial state
        :return:
        """
        raise NotImplementedError("This is an abstract task.")