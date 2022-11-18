from task_interleaving.agent.Agent import Agent
from task_interleaving.TaskPursuingFunction import TaskPursuingFunction


class QAgent(Agent):
    """
    RL agent implementing Q-learning
    """

    def __init__(self, state_value_function):
        Agent.__init__(self, state_value_function)

    def init(self, task):
        self.state = task.get_state()

    def explore(self, task):
        self.action = self.choose_epsilon_greedy_action(self.state)
        task.perform_action(self.action)

    def update(self, task, num_time_steps=None):
        """
        Attains the reward in the current task and updates the underlying state action value function according to the
        q-learning algorithm.
        :param task: task instance
        :param num_time_steps: passed time steps in the underlying SMDP
        :return:
        """
        reward = task.get_reward()
        next_state = task.get_state()
        self.update_action = self.choose_optimal_action(next_state)
        # On the lower level of the hierarchy the get_value function needs to know that it is called from update.
        if isinstance(self.function, TaskPursuingFunction):
            next_q = self.function.get_value(next_state, self.update_action, is_update=True)
        else:
            next_q = self.function.get_value(next_state, self.update_action)
        discount_factor = self.discount_factor
        #  discount gamma with number of time steps for SMDP
        if num_time_steps is not None:
            discount_factor **= num_time_steps
        td_target = reward + discount_factor * next_q
        # old_q = self.function.get_value(self.state, self.action)
        self.function.update(self.state, self.action, td_target)
        # if isinstance(self.function, TaskPursuingFunction) and self.update_action == 1 and next_state is not None:
        #     print("next_state: {} next_q: {}, direct: {}".format(next_state, task.task.task_type, next_q, self.function.function.q[next_state, self.update_action]))
        #     print(self.function.function.q[:, self.update_action])
        # if td_target != 0:
        # if td_target > 10 and reward < 10:
        #     q = self.function.get_value(self.state, self.action)
        #     print("{}".format(task.task.task_type))
        #     print("state: {}, action: {}, reward: {}, td_target: {}".format(self.state, self.action, reward, td_target))
        #     print("next_state: {}, update_action: {}, next_q: {}".format(next_state, self.update_action, next_q))
        #     print("old_q: {}, q: {}".format(old_q, q))
        #     if hasattr(self.function, 'function'):
        #         print(self.function.function.q)
        #     print()
        self.state = next_state
