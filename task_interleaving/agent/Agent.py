import numpy as np
from task_interleaving.TaskSwitchingTask import TaskSwitchingTask


class Agent:
    """
    Abstract RL agent class
    """

    def __init__(self, state_value_function):
        """
        Constructor
        :param state_value_function: instance of state value function of agent
        """
        self.function = state_value_function
        self.epsilon = 0.1
        self.discount_factor = 0.8
        self.state = None
        self.action = None
        self.update_action = None
        self.is_optimal = False

    def init(self, task):
        """
        Initializes the agent with a task instance
        :param task: instance of task
        :return:
        """
        raise NotImplementedError("This is an abstract agent.")

    def explore(self, task):
        """
        Explores the optimal action to take for the passed task instance
        :param task: task instance
        :return:
        """
        raise NotImplementedError("This is an abstract agent.")

    def update(self, task):
        """
        Attains the reward in the current task and updates the underlying state action value function accordingly.
        :param task: task instance
        :return:
        """
        raise NotImplementedError("This is an abstract agent.")

    def choose_optimal_action(self, state):
        """
        Returns the optimal action from the state value function given a state.
        :param state: state
        :return: index of optimal action
        """
        max_val = -np.inf
        max_action = -1
        actions = self.function.get_available_actions(state)
        for a in actions:
            value = self.function.get_value(state, a)
            if value > max_val or (value == max_val and np.random.random_sample() > 0.5):
                max_val = value
                max_action = a

        return max_action

    def choose_epsilon_greedy_action(self, state):
        """
        Returns the epsilon greedy action from the state value function given a state.
        :param state: state
        :return: index of optimal action
        """
        rand_num = np.random.random_sample()
        if rand_num > self.epsilon:
            self.is_optimal = True
            return self.choose_optimal_action(state)
        else:
            self.is_optimal = False
            actions = self.function.get_available_actions(state)
            index = np.random.randint(0, len(actions))
            return actions[index]

    def control(self, task):
        """
        Performs the optimal action in the passed task instance
        :param task: instance of task
        :return:
        """
        self.state = task.get_state()
        self.action = self.choose_optimal_action(self.state)
        task.perform_action(self.action)

    def random(self, task):
        """
        Performs a random action in the passed task instance
        :param task: instance of task
        :return:
        """
        self.state = task.get_state()
        actions = self.function.get_available_actions(self.state)
        index = np.random.randint(0, len(actions))
        self.action = actions[index]
        if isinstance(task, TaskSwitchingTask):
            task.perform_random_action(self.action)
        else:
            task.perform_action(self.action)