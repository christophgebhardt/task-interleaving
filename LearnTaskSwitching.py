import numpy as np
from task_interleaving.TaskSwitching import TaskSwitching
from task_interleaving.TaskPursuing import TaskPursuing
from task_interleaving.TaskSwitchingFunction import TaskSwitchingFunction
from task_interleaving.TaskSwitchingFunctionFlatRL import TaskSwitchingFunctionFlatRL
from task_interleaving.agent.QTable import QTable
from utils import get_state_trajectory


class LearnTaskSwitching:
    """
    Class managing learning of the HRL agent
    """

    def __init__(self, task_list, ** kwargs):
        """
        Constructor
        :param task_list: list of instances of TaskClass
        :param verbose_switching: flag indicating if higher level is verbose (default = 0)
        :param verbose_pursuing: flag indicating if lower level is verbose (default = false)
        :param learn_switch_costs: flag indicating if resumption costs should be considered on higher level
        (default = false)
        :param learn_pursue_costs: flag indicating if resumption costs should be considered within the task
        (default = false)
        :param learning_rate: float (default = 0.1)
        :param epsilon: float (default = 0.2)
        :param num_episodes: number of episodes in learning (default = 100000)
        :param gamma_0: Sets the discount factor of the higher level (default = 0.1)
        :param gamma_1: Sets the discount factor of the lower level (default = 0.8)
        :param reward_threshold: Expected out of task reward for learning a lower level policy only (default = 0.5)
        :param is_reward_from_obs: flag that indicates if reward value is attained from observation or function
        (default = false)
        :param is_hierarchically_optimal: flag indicating if HRL agent acts hierarchically or recursively optimal
        (default = true).
        :param is_reward_in_state: flag indicating if reward is in state feature vector (default = false)
        :param is_smdp: flag indicating if underlying decision process is an SMDP or MDP (default = false)
        :param is_flat_rl: flag indicating if task interleaving is solved with flat RL or HRL (default = false)
        :param init_functions: dictionary of instances of TaskPursuingFunction indexed by task type (default = None)
        """
        self.task_list = task_list
        self.num_tasks = len(task_list)

        # learning parameters
        self.learn_switch_costs = False
        self.learn_pursue_costs = False  # parameter to specify whether ot not to learn costs
        self.learning_rate = 0.1
        self.epsilon = 0.1  # of epsilon greedy policy
        self.gamma_0 = 0.1  # discount factor
        self.gamma_1 = 0.8  # discount factor
        self.num_episodes = 250
        # debug parameters
        self.verbose_switching = 0
        self.verbose_pursuing = False
        self.reward_threshold = 0.5
        self.is_reward_from_obs = False
        is_hierarchically_optimal = True
        self.is_smdp = False
        is_flat_rl = False
        init_functions = None

        for key, value in kwargs.items():
            if key == "learn_switch_costs":
                self.learn_switch_costs = value
            if key == "learn_pursue_costs":
                self.learn_pursue_costs = value
            elif key == "learning_rate":
                self.learning_rate = value
            elif key == "epsilon":
                self.epsilon = value
            elif key == "gamma_0":
                self.gamma_0 = value
            elif key == "gamma_1":
                self.gamma_1 = value
            elif key == "num_episodes":
                self.num_episodes = value
            elif key == "verbose_switching":
                self.verbose_switching = value
            elif key == "verbose_pursuing":
                self.verbose_pursuing = value
            elif key == "reward_threshold":
                self.reward_threshold = value
            elif key == "reward_from_observation":
                self.is_reward_from_obs = value
            elif key == "is_hierarchically_optimal":
                is_hierarchically_optimal = value
            elif key == "is_smdp":
                self.is_smdp = value
            elif key == "is_flat_rl":
                is_flat_rl = value
            elif key == "init_functions":
                init_functions = value

        # construct hierarchical experiment instance
        self.in_task_instances = self.__construct_hierarchy()

        trained_dict = None
        # if trained tables are given, initialize with them
        if init_functions is not None:
            trained_dict = init_functions[0].copy()
            for i in range(len(self.in_task_instances)):
                task_type = self.in_task_instances[i].task.task_type
                if task_type in init_functions[1].keys():
                    self.in_task_instances[i].function.function = init_functions[1][task_type]

        if is_flat_rl:
            function = TaskSwitchingFunctionFlatRL(self.in_task_instances, trained_dict)
        else:
            function = TaskSwitchingFunction(self.in_task_instances, trained_dict)

        self.experiment_instance = TaskSwitching(self.in_task_instances, function, is_hierarchically_optimal)
        self.experiment_instance.set_parameters(self.verbose_switching, self.learn_switch_costs, self.learning_rate,
                                                self.epsilon, self.num_episodes)
        self.set_gamma_0(self.gamma_0)
        self.set_gamma_1(self.gamma_1)

    def __construct_hierarchy(self):
        """
        Function constructs the hierarchy of the HRL agent by initializing all subroutines
        :return: list of instances of TaskPursuing
        """
        in_task_instances = []
        num_actions = 2
        for i in range(self.num_tasks):
            function = None
            for j in range(len(in_task_instances)):
                if self.task_list[i].task_type is not None and \
                                self.task_list[i].task_type == in_task_instances[j].task.task_type:
                    function = in_task_instances[j].function.function
                    # print("same type")
                    break

            if function is None:
                function = QTable(num_actions, self.task_list[i].task_len + 1)

            instance = TaskPursuing(self.task_list[i], self.is_reward_from_obs, function, self.is_smdp)
            instance.set_parameters(self.verbose_pursuing, self.learn_pursue_costs, self.learning_rate, self.epsilon)
            in_task_instances.append(instance)

        return in_task_instances

    def set_gamma_0(self, discount_factor):
        """
        Sets the discount factor of the higher level
        :param discount_factor: value
        :return:
        """
        self.experiment_instance.set_discount_factor(discount_factor)

    def set_gamma_1(self, discount_factor):
        """
        Sets the discount factor of the lower level
        :param discount_factor: value
        :return:
        """
        for instance in self.in_task_instances:
            instance.set_discount_factor(discount_factor)

    def learn_in_task_policy(self, task_num):
        """
        Learns the subroutine policy for the specified task
        :param task_num: number of task to learn policy for
        :return: array of the reward attained in each step of learning
        """
        exp_instance = self.in_task_instances[task_num]
        exp_instance.function.expected_out_of_task_reward = self.reward_threshold
        reward_per_step_array = np.array([])
        for i in range(self.num_episodes):
            exp_instance.reset()
            reward_per_run, steps_per_run = exp_instance.run_experiment()
            reward_per_step_array = np.append(reward_per_step_array, reward_per_run / steps_per_run)

        return reward_per_step_array

    def test_in_task_policy(self, task_num):
        """
        Executes the subroutine policy for the specified task
        :param task_num: number of task to learn policy for
        :return: tuple of attained total reward per step (array), reward per step (array), sequence of actions
        """
        exp_instance = self.in_task_instances[task_num]
        # exp_instance.verbose = False
        total_reward_at_step, reward_at_step, action_sequence = exp_instance.test_policy()
        return total_reward_at_step, reward_at_step, action_sequence

    def learn_task_switching_policy(self):
        """
        Runs the learning of the HRL agent on all levels and with all subroutines.
        :return: tuple of attained reward per episode (array) and average reward per episode (array)
        """
        return self.experiment_instance.run_experiment()

    def test_task_switching_policy(self):
        """
        Executes the learned policy considering all levels and subroutines.
        :return: tuple of attained total reward per step (array), attained reward per step (array), resumption costs per
        step (array), sequence of observations (array), task per step (array)
        """
        return self.experiment_instance.test_policy()

    def test_random_policy(self):
        """
        Executes the random policy on the task interleaving problem.
        :return: tuple of attained total reward per step (array), attained reward per step (array), resumption costs per
        step (array), sequence of observations (array), task per step (array)
        """
        return self.experiment_instance.test_policy(True)

    def test_greedy_policy(self):
        """
        Executes the greedy policy considering all levels and subroutines (in the paper it is called omniscient-myopic
        policy).
        :return: tuple of attained total reward per step (array), attained reward per step (array), resumption costs per
        step (array), sequence of observations (array), task per step (array)
        """
        return self.experiment_instance.test_greedy_policy()

    def test_against_reference_trajectory(self, ref_trajectory):
        """
        Tests the learned policy, the greedy policy and the random policy against a reference trajectory specifying the
        sequence of actions of a participant of the study
        :param ref_trajectory: sequence of actions of participant of the study
        :return: tuple of percent of matching actions of agent, random policy and greedy policy.
        """
        if self.is_smdp:
            ref_trajectory, _ = get_state_trajectory(ref_trajectory)
        return self.experiment_instance.test_ref_trajectory(ref_trajectory)

    def reset(self):
        """
        Resets the learning setting
        :return:
        """
        self.experiment_instance.function.reset()
        for i in range(len(self.in_task_instances)):
            self.in_task_instances[i].function.reset()

    @staticmethod
    def __get_subtask_num(state, task_type):
        """
        Returns the subtask of the passed state and task type.
        :param state: state representation
        :param task_type: string specifying task type
        :return: subtask number
        """
        if task_type == "calculating":
            if state < 3:
                return 0
            elif state < 6:
                return 1
            elif state < 12:
                return 2
            elif state < 18:
                return 3
            elif state < 27:
                return 4
            else:
                return 5
        elif task_type == "checking":
            if state < 5:
                return 0
            elif state < 10:
                return 1
            elif state < 15:
                return 2
            elif state < 20:
                return 3
            elif state < 25:
                return 4
            else:
                return 5
        elif task_type == "reading":
            if state < 7:
                return 0
            elif state < 12:
                return 1
            elif state < 24:
                return 2
            else:
                return 3
        else:  # typing
            if state < 4:
                return 0
            elif state < 8:
                return 1
            elif state < 12:
                return 2
            elif state < 16:
                return 3
            elif state < 20:
                return 4
            else:
                return 5
