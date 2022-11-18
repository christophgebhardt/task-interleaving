from task_interleaving.agent.StateValueFunction import StateValueFunction


class TaskPursuingFunction(StateValueFunction):
    """
    Class representing a state value function for the task pursuing level (lower) of the hierarchy. It is a wrapper
    around an instance of StateValueFunction to encapsulate additional functionality.
    """

    def __init__(self, function, environment, terminal_state, terminal_action):
        """
        Constructor
        :param function: instance of StateValueFunction
        :param environment: instance of TaskPursuingEnvironment
        :param terminal_state: vector of terminal state
        :param terminal_action: value of terminal action
        """
        StateValueFunction.__init__(self, 0)
        self.function = function
        self.terminal_state = terminal_state
        self.terminal_action = terminal_action
        self.environment = environment
        self.expected_out_of_task_reward = 0

    def get_available_actions(self, state):
        if state is self.terminal_state or self.environment.state_equal_max_state(state):
            return [1]
        else:
            return [0, 1]

    def get_value(self, state, action, is_update=False):
        """
        For hierarchical optimality get_value function on the lower level should not consider out of task rewards when
        called within the update method (communicated via the is_update flag). Tests showed that this distinction
        actually has a limited effect.
        :param state:
        :param action:
        :param is_update:
        :return:
        """
        if state is self.terminal_state:
            q_value = 0
        else:
            q_value = self.function.get_value(state, action)
            if not is_update and action == self.terminal_action:
                q_value += self.expected_out_of_task_reward

        return q_value

    def update(self, state, action, value):
        self.function.update(state, action, value)

    def reset(self):
        self.function.reset()
