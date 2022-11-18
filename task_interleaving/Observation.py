import numpy as np


class Observation:
    """
    Class represents an observation from the TaskPursuingEnvironment
    """

    def __init__(self, time, task_id, subtask_id, task_type, observation, norm_scrollbar_pos):
        """
        Constructor
        :param time: timestamp of when the observation happened
        :param task_id: id of the task (int)
        :param subtask_id: id of the subtask (int)
        :param task_type: string specifying the task type
        :param observation: observation of participant as logged in study
        :param norm_scrollbar_pos: position of scrollbar (used to compute observations in RL setting)
        """
        self.time = time
        self.task_id = task_id
        self.subtask_id = subtask_id
        self.task_type = task_type
        self.observation = observation
        self.reward = 0.0
        self.cost = 0.0
        self.is_subtask_bound = False
        self.state_estimate = -1
        self.observation_discrete = None
        self.norm_scrollbar_pos = norm_scrollbar_pos
        self.observation_continuous = np.array([float(self.subtask_id) + norm_scrollbar_pos]) #, self.reward

    def get_observation(self):
        """
        Returns an array representing the current observation
        :return: array representing current observation
        """
        return self.observation_discrete

    def set_state_estimate(self, state_estimate):
        """
        Sets the state estimate (and the discrete observation)
        :param state_estimate: setter value
        :return:
        """
        self.state_estimate = state_estimate
        self.observation_discrete = np.array([state_estimate])

    def set_observation_continuous(self, observation_continuous):
        """
        Sets the continuous observations
        :param observation_continuous: setter value
        :return:
        """
        self.observation_continuous = observation_continuous

    def add_reward(self, reward):
        """
        Updates the reward of this observation
        :param reward: setter value
        :return:
        """
        self.reward += reward
        if len(self.observation_continuous) > 1:
            self.observation_continuous[1] = self.reward

