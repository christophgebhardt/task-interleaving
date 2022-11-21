import numpy as np

from task_interleaving.agent.Environment import Environment
from task_interleaving.Observation import Observation


class TaskPursuingEnvironment(Environment):
    """
    Class represents environments on the lower level of the hierarchy, the task pursuing level. These environments
    simulate a particular user from the study by sampling from their trajectories and based on the trajectory deciding
    if the environment transitions in the next state or not.
    """

    def __init__(self, task, terminal_state, terminal_action):
        """
        Constructor
        :param task: instance of StudyTask
        :param terminal_state: vector specifying the terminal state
        :param terminal_action: value specifying the terminal action
        """
        Environment.__init__(self)
        self.current_action = None
        self.trajectories = task.trajectories
        # max state is calculated based on the max state in this task's trajectories
        self.max_state = np.zeros(self.trajectories[0][-1].get_observation().shape[0])
        state_estimate = -1
        for i in range(len(self.trajectories)):
            state = self.trajectories[i][-1].get_observation()
            if self.__state1_larger_state2(state, self.max_state):
                self.max_state = state
                state_estimate = self.trajectories[i][-1].state_estimate
        print("{} max state in traj: {} / {}".format(task.task_type, self.max_state, state_estimate))
        print("{} task length: {}".format(task.task_type, task.task_len))

        self.task_num = task.task_num
        self.terminal_state = terminal_state
        self.terminal_observation = Observation(0, task.task_num, 0, task.task_type, 0, 0)
        self.terminal_observation.observation_continuous = terminal_state
        self.terminal_observation.observation_discrete = terminal_state
        self.terminal_action = terminal_action
        self.prog_ix = 0
        if len(self.trajectories) == 1:
            self.traj_ix = 0
        else:
            self.traj_ix = np.random.randint(0, len(self.trajectories) - 1)
        self.active_trajectory = self.trajectories[self.traj_ix]
        self.observation = None
        self.__update_observation(self.active_trajectory[self.prog_ix])
        self.reward = 0

    def perform_action(self, action, progress=None):
        """
        Performs action in the environment
        :param action: action
        :param progress: allows forcing progress for a certain number of states (default=None)
        :return:
        """
        if action == self.terminal_action:
            self.observation = self.terminal_observation
        else:
            self.prog_ix += 1
            # progress variable allows forcing progress for a certain number of states
            if progress is not None and self.prog_ix < len(self.active_trajectory):
                temp_obs = self.active_trajectory[self.prog_ix].observation_discrete[0]
                obs = 0 if self.observation.observation_discrete is None else self.observation.observation_discrete[0]
                while self.prog_ix < len(self.active_trajectory) and temp_obs - obs < progress:
                    self.prog_ix += 1
                    temp_obs = self.active_trajectory[self.prog_ix].observation_discrete[0]
                # print(obs)
                # print(temp_obs)

            # check if current active trajectory still has entries if not update
            if self.prog_ix >= len(self.active_trajectory):
                active_trajectory, prog_ix, traj_ix = self.__change_active_trajectory()
                if active_trajectory is not None:
                    self.active_trajectory = active_trajectory
                    self.prog_ix = prog_ix
                    self.traj_ix = traj_ix
                else:
                    self.prog_ix = len(self.active_trajectory) - 1

            self.__update_observation(self.active_trajectory[self.prog_ix])
            self.reward = self.active_trajectory[self.prog_ix].reward

    def get_observation(self):
        """
        Returns current observation and reward of environment
        :return: tuple of instance of observation and float of reward
        """
        return self.observation, self.reward

    def reset(self):
        # reset trajectory by sampling at random
        if len(self.trajectories) == 1:
            self.traj_ix = 0
        else:
            self.traj_ix = np.random.randint(0, len(self.trajectories) - 1)
        self.active_trajectory = self.trajectories[self.traj_ix]
        # reset trajectory index
        self.prog_ix = 0
        self.current_action = 0
        # get first observation
        self.__update_observation(self.active_trajectory[self.prog_ix])

    def __change_active_trajectory(self):
        """
        Changes the trajectory from which state to state transitions are sampled
        :return: tuple of new active trajectory (list of instances of Observation), progress on trajectory, and index
        of trajectory within self.trajectories
        """
        # print("change_active_trajectory")
        for i in range(len(self.trajectories)):
            if i == self.traj_ix:
                continue
            for j in range(len(self.trajectories[i])):
                if (self.observation.get_observation() is self.terminal_state or
                        self.__state1_larger_state2(self.trajectories[i][j].get_observation(),
                                                    self.observation.get_observation())):
                    return self.trajectories[i], j, i

        return None, 0, 0

    def state_equal_max_state(self, state):
        """
        Compares if passed state equals the last state of the environment
        :param state: array representing state
        :return: flag that indicates if last state is passed state
        """
        return self.max_state[0] == state[0]

    @staticmethod
    def __state1_larger_state2(state1, state2):
        """
        Utility function comparing if state1 is later in the task than state2
        :param state1: array representing state
        :param state2: array representing state
        :return: flag that indicates if state1 is later in the task than state2
        """
        return state1[0] > state2[0]

    def __update_observation(self, observation):
        """
        Updates the current observation available in this environment
        :param observation: setter value
        :return:
        """
        self.observation = observation
        self.observation.task_id = self.task_num
