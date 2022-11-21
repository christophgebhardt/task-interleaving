import numpy as np
from copy import copy

from task_interleaving.TaskSwitchingEnvironment import TaskSwitchingEnvironment
from task_interleaving.TaskSwitchingTask import TaskSwitchingTask
from task_interleaving.agent.QAgent import QAgent
from task_interleaving.agent.SarsaAgent import SarsaAgent
from task_interleaving.task_colors import task_colors


class TaskSwitching:
    """
    Class manages the learning of an RL-agent on the higher level (task interleaving level) of the hierarchy.
    """

    def __init__(self, in_task_instances, function, is_hierarchically_optimal=True):
        """
        Constructor
        :param in_task_instances: instances of TaskPursuing
        :param function: instance of TaskSwitchingFunction
        :param is_hierarchically_optimal: flag indicating if HRL agent acts hierarchically or recursively (false)
        optimal (default = true).
        """
        # parameters
        self.verbose = 0

        self.function = function
        self.environment = TaskSwitchingEnvironment(in_task_instances, self.function)
        self.environment.is_hierarchically_optimal = is_hierarchically_optimal
        self.rl_task = TaskSwitchingTask(self.environment)
        self.agent = QAgent(self.function)
        # self.agent = SarsaAgent(self.function)
        self.num_episodes = 100000

    def run_experiment(self):
        """
        Runs the learning of the HRL agent on all levels and with all subroutines.
        :return: tuple of attained reward per episode (array) and average reward per episode (array)
        """
        episode_counter = 0
        reward_per_episode_array = np.array([])
        reward_per_episode = 0.0
        avg_reward_per_episode = np.array([])
        total_reward = 0.0
        completion = -np.inf
        self.rl_task.is_learning = True
        self.reset()

        self.agent.init(self.rl_task)
        while episode_counter < self.num_episodes:
            self.agent.explore(self.rl_task)
            state = self.agent.state
            self.agent.update(self.rl_task, self.rl_task.steps_in_task)
            reward_per_episode += self.rl_task.get_reward_stats()

            if self.verbose == 2:
                print("")
                print("[root] Number episode: {}".format(episode_counter))
                print("[root] Step of episode: {}".format(self.rl_task.num_steps))
                print("[root] Current state: {}".format(state))
                print("[root] Current action: {}".format(self.agent.action))
                print("[root] Reward: {}".format(self.rl_task.get_reward_stats()))
                print("[root] Next state: {}".format(self.agent.state))
                print("[root] Update action: {}".format(self.agent.update_action))

            if self.rl_task.is_finished():
                # reward_per_episode /= self.rl_task.num_steps # to normalize given different episode length
                self.rl_task.reset()
                self.environment.reset()  # to reset tasks after episode finished
                episode_counter += 1
                self.environment.update_target_net(episode_counter)
                reward_per_episode_array = np.append(reward_per_episode_array, reward_per_episode)
                total_reward += reward_per_episode
                avg_reward_per_episode = np.append(avg_reward_per_episode, [total_reward / episode_counter])
                if self.verbose == 2:
                    print("")
                    print("[root] Reward per episode: {}".format(reward_per_episode))
                reward_per_episode = 0.0
                # print progress
                temp = int(100 * (float(episode_counter) / float(self.num_episodes)))
                if self.verbose == 1 and temp >= completion + 10:
                    completion = temp
                    print("{}% completed".format(completion))

        return reward_per_episode_array, avg_reward_per_episode

    def test_policy(self, is_random=False):
        """
        Executes the learned policy considering all levels and subroutines.
        :param is_random: flag indicating if a randomly acting policy is executed (default = false).
        :return: tuple of attained total reward per step (array), attained reward per step (array), resumption costs per
        step (array), sequence of observations (array), task per step (array)
        """
        total_reward = 0.0
        total_reward_at_step = np.array([])
        reward_at_step = np.array([])
        costs_at_step = np.array([])
        observation_sequence = np.array([])
        task_at_step = []
        self.rl_task.is_learning = False
        self.reset()

        self.agent.init(self.rl_task)
        while not self.rl_task.is_finished():
            state = self.rl_task.get_state()
            if is_random:
                self.agent.random(self.rl_task)
            else:
                self.agent.control(self.rl_task)
            # reward = self.rl_task.get_reward()

            # self.rl_task.total_reward_in_task += total_reward - self.rl_task.resumption_costs
            self.rl_task.total_reward_in_task += total_reward
            total_reward_at_step = np.append(total_reward_at_step, self.rl_task.total_reward_in_task)
            total_reward = total_reward_at_step[-1]
            reward_at_step = np.append(reward_at_step, self.rl_task.reward_at_step_in_task)
            costs_at_step = np.append(costs_at_step, np.array([self.rl_task.resumption_costs]))
            costs_at_step = np.append(costs_at_step, np.zeros(len(self.rl_task.observation_sequence_in_task) - 1))
            observation_sequence = np.append(observation_sequence, self.rl_task.observation_sequence_in_task)
            for i in range(len(self.rl_task.observation_sequence_in_task)):
                task_at_step.append(task_colors[self.agent.action])

            if self.verbose == 2:
                print("")
                print("[root] Step of episode: {}".format(self.rl_task.num_steps))
                print("[root] Current state: {}".format(state))
                print("[root] Current action: {}".format(self.agent.action))
                print("[root] Reward: {}".format(self.rl_task.get_reward_stats()))
                # print("[root] Next state: {}".format(self.agent.state))

        return total_reward_at_step, reward_at_step, costs_at_step, observation_sequence, task_at_step

    def test_greedy_policy(self):
        """
        Executes the greedy policy considering all levels and subroutines (in the paper it is called omniscient-myopic
        policy).
        :return: tuple of attained total reward per step (array), attained reward per step (array), resumption costs per
        step (array), sequence of observations (array), task per step (array)
        """
        total_reward = 0.0
        total_reward_at_step = []
        reward_at_step = []
        costs_at_step = []
        observation_sequence = []
        task_at_step = []
        self.rl_task.is_learning = False
        self.reset()

        self.agent.init(self.rl_task)
        num_pursuing_actions = 0
        while not self.rl_task.is_finished():
            state = self.rl_task.get_state()
            action_switching, action_pursuing = self.__get_greedy_agent_action(state)
            # action_switching, action_pursuing = self.get_greedy_agent_action_reward(state)
            # action_switching, action_pursuing = self.get_greedy_agent_action_cost(state)
            self.rl_task.state = action_switching
            # select task_pursuing with state rather than action_switching
            task_pursuing = self.environment.in_task_instances[state]
            task_pursuing.rl_task.perform_action(action_pursuing)

            if action_pursuing == 0:
                task_pursuing.agent.state = task_pursuing.rl_task.get_state()
                total_reward += task_pursuing.rl_task.get_stats_reward()
                total_reward_at_step.append(total_reward)
                reward_at_step.append(task_pursuing.rl_task.get_stats_reward())
                observation_sequence.append(copy(task_pursuing.environment.observation))
                task_at_step.append(task_colors[state])
                num_pursuing_actions += 1
            else:
                if num_pursuing_actions > 0:
                    costs_at_step = np.append(costs_at_step, np.zeros(num_pursuing_actions - 1))
                # select task_pursuing with action_switching rather than state
                task_pursuing = self.environment.in_task_instances[action_switching]
                costs_at_step = np.append(costs_at_step, np.array([task_pursuing.rl_task.get_resumption_costs(
                    task_pursuing.rl_task.last_observation)]))
                num_pursuing_actions = 0

            if self.verbose == 2:
                print("")
                print("[root] Step of episode: {}".format(self.rl_task.num_steps))
                print("[root] Current state: {}".format(state))
                print("[root] Current action: {}".format(self.agent.action))
                print("[root] Reward: {}".format(self.rl_task.get_reward_stats()))
                # print("[root] Next state: {}".format(self.agent.state))

        # adjust costs_at_step to have same length as other arrays
        diff = len(reward_at_step) - len(costs_at_step)
        if diff > 0:
            costs_at_step = np.append(costs_at_step, np.zeros(diff))

        return np.array(total_reward_at_step), np.array(reward_at_step), np.array(costs_at_step), \
               np.array(observation_sequence), np.array(task_at_step)

    # calculate stats on agents decision
    def test_ref_trajectory(self, ref_trajectory):
        """
        Tests the learned policy, the greedy policy and the random policy against a reference trajectory specifying the
        sequence of actions of a participant of the study. Computes more extensive statistic than test_ref_trajectory.
        :param ref_trajectory: sequence of actions of participant of the study
        :return: tuple of extensive statistics of policies.
        """
        self.rl_task.is_learning = False
        self.reset()
        # task switching level
        num_actions = 0
        num_correct_actions = 0
        num_correct_actions_random = 0
        num_correct_actions_greedy = 0

        # task pursuing level: leave
        num_correct_actions_leave = 0
        num_correct_actions_random_leave = 0
        num_correct_actions_greedy_leave = 0

        # task pursuing level: continue
        num_actions_continue = 0
        num_correct_actions_continue = 0
        num_correct_actions_random_continue = 0
        num_correct_actions_greedy_continue = 0

        reward_in_task = {}
        time_in_task = {}

        obs = ref_trajectory[0]
        self.agent.init(self.rl_task)
        state = self.rl_task.get_state()
        task = self.agent.choose_optimal_action(state)
        num_actions += 1
        # for the first decision it is sufficient to pick the same task type
        if self.environment.in_task_instances[task].task.task_type == obs.task_type:
            num_correct_actions += 1
        # random baseline on task switching level
        rand_action = self.__get_random_switching_action(state)
        if rand_action == obs.task_id:
            num_correct_actions_random += 1
        # greedy baseline on task switching level
        greedy_action_s, _ = self.__get_greedy_agent_action(-1)
        if greedy_action_s == obs.task_id:
            num_correct_actions_greedy += 1

        for i in range(0, len(ref_trajectory) - 1):
            obs = ref_trajectory[i]
            next_obs = ref_trajectory[i + 1]
            if next_obs.task_id == obs.task_id:
                num_actions_continue += 1
            else:
                num_actions += 1

            # get state of ppt and set in_task_instance accordingly
            state = self.__get_state_of_observation(obs, reward_in_task, time_in_task)
            self.environment.in_task_instances[obs.task_id].agent.state = state

            # check decision of agent
            q_max, _ = self.environment.get_max_q_value(obs.task_id)
            self.environment.in_task_instances[obs.task_id].function.expected_out_of_task_reward = q_max
            action = self.environment.in_task_instances[obs.task_id].agent.choose_optimal_action(state)
            # action 0 is continue, action 1 is leave task
            if next_obs.task_id == obs.task_id and action == 0:
                num_correct_actions_continue += 1
            elif next_obs.task_id != obs.task_id:
                if action == 1:
                    num_correct_actions_leave += 1
                task = self.agent.choose_optimal_action(obs.task_id)
                if task == next_obs.task_id:
                    num_correct_actions += 1

            # check if greedy baseline correct
            greedy_action_s, greedy_action_p = self.__get_greedy_agent_action(obs.task_id)
            if next_obs.task_id == obs.task_id and greedy_action_p == 0:
                num_correct_actions_greedy_continue += 1
            elif next_obs.task_id != obs.task_id:
                if greedy_action_p == 1:
                    num_correct_actions_greedy_leave += 1
                if greedy_action_s == next_obs.task_id:
                    num_correct_actions_greedy += 1

            # check if random baseline correct
            rand_action = self.__get_random_pursuing_action()
            if next_obs.task_id == obs.task_id and rand_action == 0:
                num_correct_actions_random_continue += 1
            elif next_obs.task_id != obs.task_id:
                if rand_action == 1:
                    num_correct_actions_random_leave += 1
                # random baseline on task switching level
                rand_action = self.__get_random_switching_action(obs.task_id)
                if rand_action == next_obs.task_id:
                    num_correct_actions_random += 1

        # task switching level
        percent_correct = num_correct_actions / num_actions
        percent_greedy = num_correct_actions_greedy / num_actions
        percent_random = num_correct_actions_random / num_actions

        # task pursuing level: leave
        percent_correct_leave = num_correct_actions_leave / (num_actions - 1)
        percent_greedy_leave = num_correct_actions_greedy_leave / (num_actions - 1)
        percent_random_leave = num_correct_actions_random_leave / (num_actions - 1)

        # task pursuing level: continue
        percent_correct_continue = num_correct_actions_continue / num_actions_continue
        percent_greedy_continue = num_correct_actions_greedy_continue / num_actions_continue
        percent_random_continue = num_correct_actions_random_continue / num_actions_continue

        stats = np.array([percent_correct, percent_greedy, percent_random, percent_correct_leave, percent_greedy_leave,
                          percent_random_leave, percent_correct_continue, percent_greedy_continue,
                          percent_random_continue])
        return stats

    def reset(self):
        """
        Resets the TaskSwitching HRL agent
        :return:
        """
        self.rl_task.reset()
        self.environment.reset()

    def set_parameters(self, verbose, learn_costs, learning_rate, epsilon, num_episodes):
        """
        Setter function for additional parameters
        :param verbose: flag indicating if verbose
        :param learn_costs: flag indicating if resumption costs should be considered within the task
        :param learning_rate: float
        :param epsilon: float
        :param num_episodes: number of episodes in learning
        :return:
        """
        self.verbose = verbose
        self.function.alpha = learning_rate
        self.rl_task.learn_costs = learn_costs
        self.agent.epsilon = epsilon  # of epsilon greedy policy
        self.num_episodes = num_episodes

    def set_discount_factor(self, discount_factor):
        """
        Setter function for the discount factor
        :param discount_factor: float
        :return:
        """
        self.agent.discount_factor = discount_factor

    def __get_state_of_observation(self, observation, reward_in_task, time_in_task):
        """
        Returns state representation of observation instance
        :param observation: instance of Observation
        :param reward_in_task: array with in task rewards
        :param time_in_task: array with in task times
        :return:
        """
        # discrete case
        if self.function.num_state_features is None:
            return observation.observation_discrete
        # continuous case
        else:
            # scroll value in state space
            if self.function.num_state_features == 1:
                return np.array([float(observation.subtask_id) + observation.norm_scrollbar_pos])
            else:
                # scroll value and reward in state space
                if observation.task_id not in reward_in_task:
                    reward_in_task[observation.task_id] = {}
                if observation.subtask_id not in reward_in_task[observation.task_id]:
                    reward_in_task[observation.task_id][observation.subtask_id] = 0

                reward_in_task[observation.task_id][observation.subtask_id] = observation.reward
                reward = 0.0
                for _, rew in reward_in_task[observation.task_id].items():
                    reward += rew

                if self.function.num_state_features == 2:
                    return np.array([float(observation.subtask_id) + observation.norm_scrollbar_pos, reward])
                else:
                    # scroll value, reward and time in state space
                    if observation.task_id not in time_in_task:
                        time_in_task[observation.task_id] = 0.0
                    state = np.array([float(observation.subtask_id) + observation.norm_scrollbar_pos, reward, time_in_task[observation.task_id]])
                    time_in_task[observation.task_id] += 0.1
                    return state

    def __get_random_switching_action(self, state):
        """
        Chooses a random actions of the once available in the passed state
        :param state: state representation
        :return: chosen action
        """
        actions = self.function.get_available_actions(state)
        index = np.random.randint(0, len(actions))
        return actions[index]

    def __get_greedy_agent_action(self, current_task):
        """
        Chooses a greedy action given the passed task
        :param current_task: task id
        :return: tuple of chosen task id and pursuing action (0 if switching_action = current_task, else 1)
        """
        max_reward = -np.inf
        task_ix = -1
        is_current_task_finished = False
        task_pursuing = self.environment.in_task_instances[current_task]
        leaving_costs = 0
        if not task_pursuing.is_finished():
            next_state = self.__estimate_next_state(task_pursuing)
            leaving_costs = task_pursuing.task.resumption[next_state]
        for i in range(len(self.environment.in_task_instances)):
            task_pursuing = self.environment.in_task_instances[i]
            if task_pursuing.is_finished():
                if i == current_task:
                    is_current_task_finished = True
                continue
            next_state = self.__estimate_next_state(task_pursuing)
            if next_state < len(task_pursuing.task.reward):
                next_reward = task_pursuing.task.reward[next_state]
                if i != current_task:
                    next_reward -= task_pursuing.task.resumption[next_state]
                    next_reward -= leaving_costs
                if next_reward > max_reward:
                    task_ix = i
                    max_reward = next_reward

        if task_ix == -1:
            if is_current_task_finished:
                new_task = current_task
                while new_task == current_task:
                    new_task = np.random.randint(0, len(self.environment.in_task_instances))
                switching_action = new_task
                pursuing_action = 1
            else:
                switching_action = current_task
                pursuing_action = 0
        else:
            switching_action = task_ix
            if task_ix == current_task:
                pursuing_action = 0
            else:
                pursuing_action = 1

        return switching_action, pursuing_action

    def __get_greedy_agent_action_reward(self, current_task):
        """
        Chooses a greedy action given the passed task only considering rewards.
        :param current_task: task id
        :return: tuple of chosen task id and pursuing action (0 if switching_action = current_task, else 1)
        """
        max_reward = -np.inf
        task_ix = -1
        is_current_task_finished = False
        task_pursuing = self.environment.in_task_instances[current_task]
        for i in range(len(self.environment.in_task_instances)):
            task_pursuing = self.environment.in_task_instances[i]
            if task_pursuing.is_finished():
                if i == current_task:
                    is_current_task_finished = True
                continue
            next_state = self.__estimate_next_state(task_pursuing)
            if next_state < len(task_pursuing.task.reward):
                next_reward = task_pursuing.task.reward[next_state]
                if next_reward > max_reward:
                    task_ix = i
                    max_reward = next_reward

        if task_ix == -1:
            if is_current_task_finished:
                new_task = current_task
                while new_task == current_task:
                    new_task = np.random.randint(0, len(self.environment.in_task_instances))
                switching_action = new_task
                pursuing_action = 1
            else:
                switching_action = current_task
                pursuing_action = 0
        else:
            switching_action = task_ix
            if task_ix == current_task:
                pursuing_action = 0
            else:
                pursuing_action = 1

        return switching_action, pursuing_action

    def __get_greedy_agent_action_cost(self, current_task):
        """
        Chooses a greedy action given the passed task only considering costs.
        :param current_task: task id
        :return: tuple of chosen task id and pursuing action (0 if switching_action = current_task, else 1)
        """
        max_reward = -np.inf
        task_ix = -1
        is_current_task_finished = False
        task_pursuing = self.environment.in_task_instances[current_task]
        leaving_costs = 0
        if not task_pursuing.is_finished():
            next_state = self.__estimate_next_state(task_pursuing)
            leaving_costs = task_pursuing.task.resumption[next_state]
        for i in range(len(self.environment.in_task_instances)):
            task_pursuing = self.environment.in_task_instances[i]
            if task_pursuing.is_finished():
                if i == current_task:
                    is_current_task_finished = True
                continue
            next_state = self.__estimate_next_state(task_pursuing)
            if next_state < len(task_pursuing.task.reward):
                next_reward = 0
                if i != current_task:
                    next_reward -= task_pursuing.task.resumption[next_state]
                    next_reward -= leaving_costs
                if next_reward > max_reward:
                    task_ix = i
                    max_reward = next_reward

        if task_ix == -1:
            if is_current_task_finished:
                new_task = current_task
                while new_task == current_task:
                    new_task = np.random.randint(0, len(self.environment.in_task_instances))
                switching_action = new_task
                pursuing_action = 1
            else:
                switching_action = current_task
                pursuing_action = 0
        else:
            switching_action = task_ix
            if task_ix == current_task:
                pursuing_action = 0
            else:
                pursuing_action = 1

        return switching_action, pursuing_action

    @staticmethod
    def __estimate_next_state(task_pursuing):
        """
        Estimates the next state for the greedy agent.
        :param task_pursuing: instance of TaskPursuing
        :return:
        """
        return task_pursuing.agent.state + 1

    @staticmethod
    def __get_random_pursuing_action():
        """
        Returns a random action for the agent on the task pursuing level. Thus, it considers the base rate of
        participants of the study in how often they chose to continue within a task.
        :return: random action
        """
        is_base_rate = False
        if is_base_rate:
            base_rate_continue = 0.95293287050656
            fract = np.random.uniform()
            if fract > base_rate_continue:
                rand_action = 1
            else:
                rand_action = 0
        else:
            rand_action = np.random.randint(2)
        return rand_action


