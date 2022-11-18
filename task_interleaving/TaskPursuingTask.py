import random
from task_interleaving.agent.Task import Task


class TaskPursuingTask(Task):
    """
    Class represents RL task of the lower level of the hierarchy (abstraction of environment with more functionality).
    """

    def __init__(self, environment, task, is_reward_from_obs=False, num_state_features=1):
        """
        Constructor
        :param environment: instance of TaskPursuingEnvironment
        :param task: instance of TaskClass
        :param is_reward_from_obs: flag that indicates if reward value is attained from observation or function
        :param num_state_features: length of the state vector (default=1)
        """
        Task.__init__(self, environment)
        self.task = task
        self.is_reward_from_obs = is_reward_from_obs
        if num_state_features > 1:
            self.reward_factors = task.reward_factors
        else:
            self.reward_factors = None
        self.learn_costs = False
        self.reward = 0
        self.last_observation, _ = self.environment.get_observation()
        self.current_state = self.last_observation.get_observation()
        self.num_time_steps = 1

    def perform_action(self, action, progress=None):
        """
        Performs a chosen action on the underlying environment. Computes the reward itself or attains it from the
        observation depending on self.is_reward_from_obs.
        :param action: the action chosen by the agent to be performed on the environment
        :param progress: allows forcing progress for a certain number of states
        """
        self.environment.perform_action(action, progress)
        observation, reward_from_obs = self.environment.get_observation()
        self.current_state = observation.get_observation()
        self.num_time_steps = observation.time
        if self.is_reward_from_obs:
            self.reward = reward_from_obs
        else:
            self.reward = 0.0
            if self.current_state is not self.environment.terminal_state and \
                            observation.state_estimate > self.last_observation.state_estimate:
                # add reward for all non-visited states between old and new progress (participant can't skip states)
                # gives reward when ENTERING rewarding state
                # for i in range(self.last_observation.state_estimate + 1, observation.state_estimate + 1):
                # gives reward when LEAVING rewarding state
                for i in range(self.last_observation.state_estimate, observation.state_estimate):
                    # in case of non discrete setting take reward factors into account
                    if self.reward_factors is not None:
                        reward_probability = self.reward_factors[observation.subtask_id]
                        rand_num = random.random()
                        if rand_num < reward_probability:
                            self.reward += self.task.reward[i]
                    else:
                        self.reward += self.task.reward[i]
                # add reward to observation
                if self.reward_factors is not None:
                    observation.add_reward(self.reward)

            self.reward = self.__calculate_reward(action, self.reward)

        if self.current_state is not self.environment.terminal_state:
            self.last_observation = observation

    def get_reward(self):
        return self.reward

    def get_stats_reward(self):
        """
        Returns rewards for analysis (lower limit is zero).
        :return: value
        """
        if self.reward < 0:
            return 0
        return self.reward

    def get_state(self):
        return self.current_state

    def is_finished(self, action=0):
        return action == self.environment.terminal_action

    def get_resumption_costs(self, observation):
        """
        Returns the resumption costs of the passed observation
        :param observation: instance of Observation
        :return: float value
        """
        cost = self.task.resumption[observation.state_estimate]
        return cost

    def reset(self):
        self.last_observation, _ = self.environment.get_observation()
        self.current_state = self.last_observation.get_observation()
        self.num_time_steps = 1

    def __calculate_reward(self, action, reward):
        """
        Adjusts the reward depending on the chosen action
        :param action: action value
        :param reward: reward value
        :return: adjusted reward value
        """
        if action == self.environment.terminal_action:
            reward = 0.0
            if self.learn_costs:
                reward -= self.get_resumption_costs(self.last_observation)

        return reward
