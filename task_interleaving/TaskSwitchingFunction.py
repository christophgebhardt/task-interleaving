import numpy as np

from task_interleaving.agent.QTable import QTable


class TaskSwitchingFunction(QTable):
    """
    Class representing a state value function for the task switching level (higher) of the hierarchy. It is a wrapper
    around an instance of QTable to encapsulate additional functionality.
    """

    def __init__(self, in_task_instances, trained_dict=None):
        """
        Constructor
        :param in_task_instances: instances of TaskPursuing
        :param trained_dict: dictionary with state value functions of higher level (optional, default=None)
        """
        QTable.__init__(self, 1, 1)
        self.in_task_instances = in_task_instances
        self.functions = []
        if trained_dict is not None:
            self.function_dict = trained_dict
        else:
            self.function_dict = {}
        self.num_tasks = len(self.in_task_instances)
        for i in range(self.num_tasks):
            task_type = self.in_task_instances[i].task.task_type
            if task_type not in self.function_dict:
                task_len = self.in_task_instances[i].task.task_len
                function = QTable(1, task_len)
                self.function_dict[task_type] = function
            else:
                function = self.function_dict[task_type]
            self.functions.append(function)

    def get_available_actions(self, state):
        # get the row of rewards for the current action
        available_actions = np.array([], dtype=int)
        for i in range(self.num_tasks):
            # if in start state or if task is not task of current state and task is not finished
            if (state == -1 or i != state) and not self.in_task_instances[i].is_finished():
                available_actions = np.append(available_actions, [i])

        # if all other tasks are finished and current task is not, add current task as available action
        if len(available_actions) == 0 and not self.in_task_instances[state].is_finished():
            available_actions = np.append(available_actions, [state])

        # if still no available action in array, select at random
        if len(available_actions) == 0:
            available_actions = np.append(available_actions, [np.random.randint(0, self.num_tasks)])

        return available_actions

    def update(self, state, action, value):
        in_task_state = self.in_task_instances[action].get_initial_state()
        self.functions[action].update(in_task_state, 0, value)

    def get_value(self, state, action):
        in_task_state = self.in_task_instances[action].get_state()
        return self.functions[action].get_value(in_task_state, 0)

    def reset(self):
        for function in self.functions:
            function.reset()
