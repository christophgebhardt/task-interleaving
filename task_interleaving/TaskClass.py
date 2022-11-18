class TaskClass:
    """
    Class representing the task in the task interleaving scenario. Among other things, it specifies the cost and reward
    function of a specific interleaving task.
    """

    def __init__(self, reward_array, resumption_array, task_len, task_no):
        """
        Constructor
        :param reward_array: Array specifying the reward function per state of task
        :param resumption_array: Array specifying the resumption cost per state of task
        :param task_len: length of the task
        :param task_no: id of the task
        """
        self.reward = reward_array
        self.resumption = resumption_array
        self.reward_spline = []
        self.cost_spline = []
        self.task_len = task_len
        self.task_num = task_no
        self.is_continuing = True
        self.can_leap = False
        self.var_reward_cost = 0
        self.mean_action_time = 1
        self.var_action_time = 0
        self.task_type = None
        self.run_task_once = 1

    def get_task_avg_reward(self):
        """
        Returns the average reward attainable within this task
        :return: float
        """
        cost = 0
        if self.is_continuing:
            for i in range(self.task_len):
                cost += self.resumption[i]
            cost /= self.task_len
        else:
            cost = self.resumption

        avg_reward = cost
        for i in range(self.task_len):
            avg_reward += self.reward[i]

        avg_reward /= self.task_len
        return avg_reward
