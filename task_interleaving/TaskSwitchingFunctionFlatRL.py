from task_interleaving.agent.QTable import QTable
from task_interleaving.TaskSwitchingFunction import TaskSwitchingFunction


class TaskSwitchingFunctionFlatRL(TaskSwitchingFunction):
    """
    Class representing a state value function for the task switching with the flat RL baseline.
    """

    def __init__(self, in_task_instances, trained_dict=None):
        """
        Constructor
        :param in_task_instances: instances of TaskPursuing
        :param trained_dict: dictionary with state value functions of higher level (optional, default=None)
        """
        TaskSwitchingFunction.__init__(self, in_task_instances)
        self.functions = []
        if trained_dict is not None:
            self.function_dict = trained_dict
        else:
            self.function_dict = {}
        for i in range(self.num_tasks):
            task_type = self.in_task_instances[i].task.task_type
            if task_type not in self.function_dict:
                self.function_dict[task_type] = {}
            task_len = self.in_task_instances[i].task.task_len
            for j in range(self.num_tasks):
                task_type_2 = self.in_task_instances[j].task.task_type
                if task_type_2 not in self.function_dict[task_type]:
                    task_len_2 = self.in_task_instances[j].task.task_len
                    function = QTable(task_len_2, task_len)
                    self.function_dict[task_type][task_type_2] = function

            function_array = []
            for j in range(self.num_tasks):
                task_type_2 = self.in_task_instances[j].task.task_type
                function_array.append(self.function_dict[task_type][task_type_2])
            self.functions.append(function_array)

    def update(self, state, action, value):
        in_task_state_action = self.in_task_instances[action].get_initial_state()
        in_task_state_state = self.in_task_instances[state].get_initial_state()
        if in_task_state_state is None:
            in_task_state_state = self.in_task_instances[state].get_state()
        self.functions[state][action].update(in_task_state_state, in_task_state_action, value)

    def get_value(self, state, action):
        in_task_state_action = self.in_task_instances[action].get_state()
        in_task_state_state = self.in_task_instances[state].get_state()
        return self.functions[state][action].get_value(in_task_state_state, in_task_state_action)

    def reset(self):
        for function_array in self.functions:
            for function in function_array:
                function.reset()
