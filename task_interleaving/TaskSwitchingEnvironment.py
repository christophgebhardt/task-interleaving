import numpy as np

from task_interleaving.agent.Environment import Environment


class TaskSwitchingEnvironment(Environment):
    """
    Class represents environments on the higher level of the hierarchy for task interleaving.
    """

    def __init__(self, in_task_instances, function):
        """
        Constructor
        :param in_task_instances: instances of TaskPursuing
        :param function: instance of TaskSwitchingFunction
        """
        Environment.__init__(self)
        self.in_task_instances = in_task_instances
        self.function = function
        self.num_tasks = len(self.in_task_instances)
        self.task = -1
        self.is_hierarchically_optimal = True

    def perform_action(self, action):
        self.task = action

    def get_observation(self, is_learning=True):
        q_max = 0
        if self.is_hierarchically_optimal:
            q_max, _ = self.get_max_q_value(self.task)
        if is_learning:
            return self.in_task_instances[self.task].execute_learning(q_max)
        else:
            return self.in_task_instances[self.task].execute_policy(q_max)

    def reset(self):
        # reset tasks
        for i in range(self.num_tasks):
            # reset state of task
            self.in_task_instances[i].reset()

    def get_max_q_value(self, outside_task):
        """
        Returns the highest q value available on the task interleaving level outside of outside_task
        :param outside_task: task outside of which the highest q-value is searched
        :return: tuple of highest available q value and subroutine where it is to be found
        """
        q_alt = -np.Inf
        best_action = -1
        for i in range(self.num_tasks):
            if i == outside_task or self.in_task_instances[i].is_finished():
                continue
            value = self.function.get_value(0, i)
            if value > q_alt:
                q_alt = value
                best_action = i
        if best_action == -1:
            return 0, None
        else:
            return q_alt, best_action

    def update_target_net(self, episode_num):
        """
        Updates the target net for Deep-Q-learning lower level agents
        :param episode_num: number of current episode
        :return:
        """
        for instance in self.in_task_instances:
            if hasattr(instance.function.function, 'update_target_net'):
                instance.function.function.update_target_net(episode_num)


