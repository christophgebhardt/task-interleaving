from task_interleaving.TaskClass import TaskClass


class StudyTask(TaskClass):
    """
    Specialization of TaskClass for tasks of our study
    """

    def __init__(self, reward_array, resumption_array, task_len, task_no):
        """
        Constructor
        :param reward_array: Array specifying the reward function per state of task
        :param resumption_array: Array specifying the resumption cost per state of task
        :param task_len: length of the task
        :param task_no: id of the task
        """
        TaskClass.__init__(self, reward_array, resumption_array, task_len, task_no)
        self.reward_observations = None
        self.cost_observations = None
        self.trajectories = None
        self.delta_state_distribution = None
        self.observation_state_mapping = None
        self.observation_len = 0
        self.subtask_bounds = None
