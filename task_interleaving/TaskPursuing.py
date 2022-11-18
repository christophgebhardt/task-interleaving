import numpy as np
from copy import copy

from task_interleaving.TaskPursuingTask import TaskPursuingTask
from task_interleaving.TaskPursuingFunction import TaskPursuingFunction
from task_interleaving.TaskPursuingEnvironment import TaskPursuingEnvironment
from task_interleaving.agent.SarsaAgent import SarsaAgent
from task_interleaving.agent.QAgent import QAgent


class TaskPursuing:
    """
    Class manages the learning of an RL-agent on the lower level (task pursuing level) of the hierarchy.
    """

    def __init__(self, task, is_reward_from_obs, function, is_smdp):
        """
        Constructor
        :param task: instance of TaskClass
        :param is_reward_from_obs: flag that indicates if reward value is attained from observation or function
        :param function: instance of StateValueFunction
        :param is_smdp: flag indicating if underlying decision process is an SMDP (true) or MDP (false)
        """
        self.task = task
        self.verbose = False
        self.is_smdp = is_smdp
        self.terminal_state = None
        terminal_action = 1
        num_state_features = 1
        if hasattr(function, "num_state_features"):
            num_state_features = function.num_state_features

        self.environment = TaskPursuingEnvironment(task, self.terminal_state, terminal_action)
        self.rl_task = TaskPursuingTask(self.environment, task, is_reward_from_obs, num_state_features)
        self.function = TaskPursuingFunction(function, self.environment, self.terminal_state, terminal_action)
        self.agent = QAgent(self.function)
        # self.agent = SarsaAgent(self.function)
        self.agent.init(self.rl_task)
        self.last_state = None
        self.initial_state = None

    def execute_learning(self, best_q_alt):
        """
        Executes learning in particular lower-level subroutine
        :param best_q_alt: highest available q-value outside of this task
        :return: tuple of q-value at initial state, attained reward in subroutine, resumption cost, steps executed in
        subroutine
        """
        self.initial_state = self.agent.state
        initial_cost = self.rl_task.get_resumption_costs(self.rl_task.last_observation)
        self.function.expected_out_of_task_reward = best_q_alt
        total_reward, steps_per_run = self.run_experiment()
        self.__update_state_action()
        q_task = self.function.get_value(self.initial_state, 0)
        return q_task, total_reward, initial_cost, steps_per_run

    def execute_policy(self, best_q_alt):
        """
        Executes learned policy in lower-level subroutine
        :param best_q_alt: highest available q-value outside of this task
        :return: tuple of attained total reward per step (array), attained reward per step (array), resumption costs,
        sequence of observations (array)
        """
        resumption_costs = self.rl_task.get_resumption_costs(self.rl_task.last_observation)
        self.function.expected_out_of_task_reward = best_q_alt
        total_reward_at_step, reward_at_step, observation_sequence = self.test_policy()
        self.__update_state_action()
        return total_reward_at_step, reward_at_step, resumption_costs, observation_sequence

    def execute_random_policy(self):
        """
        Executes a policy that choses actions at random in lower-level subroutine
        :return: tuple of attained total reward per step (array), attained reward per step (array), resumption costs,
        sequence of observations (array)
        """
        resumption_costs = self.rl_task.get_resumption_costs(self.rl_task.last_observation)
        total_reward_at_step, reward_at_step, observation_sequence = self.test_policy(is_random=True)
        self.__update_state_action()
        return total_reward_at_step, reward_at_step, resumption_costs, observation_sequence

    def run_experiment(self):
        """
        Learns a policy for the respective subroutine without considering other subroutines or a higher level
        :return: tuple of total reward and number of steps per run
        """
        # counter for number of runs
        steps_per_run = 0
        total_reward = 0.0

        self.agent.init(self.rl_task)
        while not self.rl_task.is_finished(self.agent.action):
            self.last_state = self.agent.state
            last_observation = self.environment.observation.get_observation()
            self.agent.explore(self.rl_task)
            num_time_steps = None
            if self.is_smdp:
                num_time_steps = self.rl_task.num_time_steps
            self.agent.update(self.rl_task, num_time_steps)
            if self.agent.is_optimal:
                total_reward += self.rl_task.get_reward()
            steps_per_run += 1
            if self.verbose:
                print("")
                print("{}".format(self.task.task_type))
                print("[{}] Step in run: {}".format(self.task.task_num, steps_per_run))
                print("[{}] Current observation: {}".format(self.task.task_num, last_observation))
                print("[{}] Current state: {}/{}".format(self.task.task_num, self.last_state, self.environment.max_state))
                print("[{}] Current action: {}".format(self.task.task_num, self.agent.action))
                print("[{}] Reward: {}".format(self.task.task_num, self.rl_task.get_reward()))
                print("[{}] Next observation: {}".format(self.task.task_num, self.environment.observation.get_observation()))
                print("[{}] Next state: {}".format(self.task.task_num, self.agent.state))
                print("[{}] Update action: {}".format(self.task.task_num, self.agent.update_action))

        return total_reward, steps_per_run

    def test_policy(self, is_random=False):
        """
        Executes a learned policy on the respective subroutine without considering other subroutines or a higher level
        :param is_random: flag that indicates if actions are taken at random (true) or according to policy
        (false, default = false)
        :return: tuple of attained total reward per step (array), attained reward per step (array), sequence of
        observations (array)
        """
        total_reward = 0.0
        total_reward_at_step = []
        reward_at_step = []
        observation_sequence = []
        steps_per_run = 0

        self.agent.init(self.rl_task)
        # force task agent to make progress in order to ensure that agent
        # can't leave task without making at least one state progress
        self.rl_task.perform_action(0, 1)
        while not self.rl_task.is_finished(self.agent.action):
            self.last_state = self.rl_task.get_state()
            if is_random:
                self.agent.random(self.rl_task)
            else:
                self.agent.control(self.rl_task)
            total_reward += self.rl_task.get_stats_reward()
            total_reward_at_step.append(total_reward)
            reward_at_step.append(self.rl_task.get_stats_reward())
            # observation_sequence.append(str(self.task.task_num) + "," + str(self.agent.state))
            observation_sequence.append(copy(self.environment.observation))
            if self.verbose:
                print("")
                print("{}".format(self.task.task_type))
                print("[{}] Step in run: {}".format(self.task.task_num, steps_per_run))
                print("[{}] Current observation: {}".format(self.task.task_num, self.environment.observation.state_estimate))
                print("[{}] Current state: {}/{}".format(self.task.task_num, self.last_state, self.environment.max_state))
                print("[{}] Current action: {}".format(self.task.task_num, self.agent.action))
                print("[{}] Reward: {}".format(self.task.task_num, self.rl_task.get_reward()))
            steps_per_run += 1

        return np.array(total_reward_at_step), np.array(reward_at_step), np.array(observation_sequence)

    def set_parameters(self, verbose, learn_costs, learning_rate, epsilon):
        """
        Setter function to set additional parameters
        :param verbose: flag indicating if verbose
        :param learn_costs: flag indicating if resumption costs should be considered within the task
        :param learning_rate: float
        :param epsilon: float
        :return:
        """
        self.verbose = verbose
        self.function.alpha = learning_rate
        self.rl_task.learn_costs = learn_costs
        self.agent.epsilon = epsilon  # of epsilon greedy policy

    def set_discount_factor(self, discount_factor):
        """
        Setter function for the discount factor
        :param discount_factor: float
        :return:
        """
        self.agent.discount_factor = discount_factor

    def is_finished(self):
        """
        Returns if task is finished
        :return: bool
        """
        # None is terminal state
        # return self.agent.state is self.terminal_state or self.environment.state_equal_max_state(self.agent.state)
        return self.environment.state_equal_max_state(self.agent.state)

    def get_state(self):
        """
        Returns current state of the task
        :return: state vector
        """
        return self.agent.state

    def get_initial_state(self):
        """
        Returns the initial state of the task
        :return: state vector
        """
        return self.initial_state

    def __update_state_action(self):
        """
        Updates state to last state and action to continue. This is necessary to do when leaving the task as otherwise
        the current state and action of a subtask would point on the leaving state and action.
        :return:
        """
        self.rl_task.current_state = self.last_state
        self.agent.state = self.last_state
        self.agent.action = 0

    def reset(self):
        """
        Resets the task
        :return:
        """
        self.environment.reset()
        self.rl_task.reset()
        self.agent.init(self.rl_task)
        self.agent.action = 0
        self.last_state = None
