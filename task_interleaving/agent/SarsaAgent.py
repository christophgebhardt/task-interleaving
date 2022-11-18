from task_interleaving.agent.Agent import Agent
from task_interleaving.TaskPursuingFunction import TaskPursuingFunction


class SarsaAgent(Agent):
    """
    RL agent implementing SARSA
    """

    def __init__(self, state_value_function):
        Agent.__init__(self, state_value_function)
        self.next_action = 0

    def init(self, task):
        self.state = task.get_state()
        # regular way to implement SarsaAgent causes bugs as tasks can be immediately left. To prevent this from
        # happening the following line is commented
        # self.action = self.choose_epsilon_greedy_action(self.state)
        self.action = 0
        self.next_action = 0

    def explore(self, task):
        self.action = self.next_action
        task.perform_action(self.action)

    def update(self, task, num_time_steps=None):
        """
        Attains the reward in the current task and updates the underlying state action value function according to the
        SARSA algorithm.
        :param task: task instance
        :param num_time_steps: passed time steps in the underlying SMDP
        :return:
        """
        reward = task.get_reward()
        next_state = task.get_state()
        self.next_action = self.choose_epsilon_greedy_action(next_state)
        self.update_action = self.next_action
        if isinstance(self.function, TaskPursuingFunction):
            next_q = self.function.get_value(next_state, self.update_action, is_update=True)
        else:
            next_q = self.function.get_value(next_state, self.update_action)
        discount_factor = self.discount_factor
        #  discount gamma with number of time steps for SMDP
        if num_time_steps is not None:
            discount_factor **= num_time_steps
        td_target = reward + self.discount_factor * next_q
        self.function.update(self.state, self.action, td_target)
        self.state = next_state
        return reward






