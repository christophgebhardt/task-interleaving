from task_interleaving.agent.Task import Task


class TaskSwitchingTask(Task):
    """
    Class represents RL task of the higher level of the hierarchy (abstraction of environment with more functionality).
    """

    def __init__(self, environment):
        """
        Constructor
        :param environment: instance of TaskSwitchingEnvironment
        """
        Task.__init__(self, environment)
        self.learn_costs = False
        self.num_steps = 0
        self.reward = 0
        self.reward_stats = 0
        self.steps_in_task = 0
        self.is_learning = True
        self.total_reward_in_task = None
        self.reward_at_step_in_task = None
        self.resumption_costs = None
        self.observation_sequence_in_task = None
        self.state = 0

    def get_reward(self):
        return self.reward

    def get_reward_stats(self):
        """
        Returns rewards for analysis.
        :return: value
        """
        return self.reward_stats

    def perform_action(self, action):
        """
        Executes learning or control on the subroutine specified by the passed action.
        :param action: value indicating subroutine
        :return:
        """
        if self.is_learning:
            self.environment.perform_action(action)
            expected_reward, reward, resumption_costs, steps_in_task = self.environment.get_observation(self.is_learning)
            self.reward_stats = reward
            self.reward = expected_reward
            if self.learn_costs:
                self.reward -= resumption_costs
        else:
            self.environment.perform_action(action)
            total_reward_in_task, reward_at_step_in_task, resumption_costs, observation_sequence_in_task = \
                self.environment.get_observation(self.is_learning)
            steps_in_task = len(observation_sequence_in_task)
            self.reward = total_reward_in_task[-1]
            self.total_reward_in_task = total_reward_in_task
            self.reward_at_step_in_task = reward_at_step_in_task
            self.resumption_costs = resumption_costs
            self.observation_sequence_in_task = observation_sequence_in_task

        self.num_steps += steps_in_task
        self.steps_in_task = steps_in_task
        self.state = action

    def perform_random_action(self, action):
        """
        Executes random policy in the subroutine specified by the passed action.
        :param action: value indicating subroutine
        :return:
        """
        self.environment.perform_action(action)
        total_reward_in_task, reward_at_step_in_task, resumption_costs, observation_sequence_in_task = \
            self.environment.in_task_instances[action].execute_random_policy()
        steps_in_task = len(observation_sequence_in_task)
        self.reward = total_reward_in_task[-1]
        self.total_reward_in_task = total_reward_in_task
        self.reward_at_step_in_task = reward_at_step_in_task
        self.resumption_costs = resumption_costs
        self.observation_sequence_in_task = observation_sequence_in_task

        self.num_steps += steps_in_task
        self.steps_in_task = steps_in_task
        self.state = action

    def get_state(self):
        return self.state

    def is_finished(self):
        """
        Indicates if episode is finished. This is the case if all subtasks have been completed.
        :return:
        """
        for in_task_instance in self.environment.in_task_instances:
            if not in_task_instance.is_finished():
                # print("not finished {} {}".format(in_task_instance.task.task_type, in_task_instance.task.task_num))
                return False
        return True

    def reset(self):
        self.num_steps = 0
        self.steps_in_task = 0