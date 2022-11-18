import numpy as np
import pickle
from LearnTaskSwitching import LearnTaskSwitching
from plot_utilities import plot_reward_learning, plot_policy_evaluation, plot_cost_to_go

from utils import update_init_tables, sparse_output


def main():
    participant_name = "0B8VVB89"
    # files
    task_file = "data/environments/" + participant_name + "_train_env.pkl"
    trajectory_file = 'data/trajectories/' + participant_name + '_train.pkl'
    state_distribution_file = 'data/distributions/' + participant_name + '_state.pkl'

    # participant trajectory and environment files for fitting
    # test_trajectory_file = "data/test-trajectories/" + participant_name + "_fit.pkl"
    # test_file = "data/environments/" + participant_name + "_fit_env.pkl"

    # participant trajectory and environment files for testing
    test_trajectory_file = "data/test-trajectories/" + participant_name + "_test.pkl"
    test_file = "data/environments/" + participant_name + "_test_env.pkl"

    # execution control
    verbose_switching = 1

    # learning parameters
    num_episodes = 600
    steps_per_episode = 3000
    learn_pursue_costs = True
    learn_switch_costs = True
    is_reward_from_obs = True
    is_smdp = True

    colors = ['b', 'g', 'm', 'c', 'r', 'y', 'maroon', 'springgreen', 'indigo', 'cornflowerblue', 'fuchsia',
              'lightpink', 'crimson', 'firebrick', 'lawngreen', 'cadetblue', 'aquamarine', 'lightsteelblue',
              'brown', 'orange', 'gold', 'purple', 'pink', 'skyblue', 'k', 'w', 'gray']

    with open(task_file, 'rb') as f:
        task_list = pickle.load(f)
    with open(state_distribution_file, 'rb') as f:
        distribution_list = pickle.load(f)
    add_state_distribution_to_task(task_list, distribution_list)
    add_trajectories_to_task(task_list, trajectory_file)

    learning_manager = LearnTaskSwitching(task_list, num_episodes=num_episodes, steps_per_episode=steps_per_episode,
                                          verbose_switching=verbose_switching, learn_pursue_costs=learn_pursue_costs,
                                          learn_switch_costs=learn_switch_costs, task_colors=colors,
                                          is_reward_from_obs=is_reward_from_obs)
    # learn task switching policy
    reward_per_episode, avg_reward_per_episode = learning_manager.learn_task_switching_policy()
    plot_reward_learning(reward_per_episode, expected_reward_array=avg_reward_per_episode, plot_title="learning curve")

    testing_manager = learning_manager
    with open(test_file, 'rb') as f:
        test_task_list = pickle.load(f)

    add_trajectories_to_task(test_task_list, test_trajectory_file)
    add_state_distribution_to_task(test_task_list, distribution_list)

    # update init tables for initializing test manager
    init_tables = update_init_tables(learning_manager)
    testing_manager = LearnTaskSwitching(test_task_list, steps_per_episode=steps_per_episode,
                                         verbose_switching=verbose_switching, task_colors=colors,
                                         is_smdp=is_smdp, init_functions=init_tables)
    type_list = []
    # evaluate in-task policies or turn off in-task prints
    for i in range(testing_manager.experiment_instance.environment.num_tasks):
        in_task_instance = testing_manager.experiment_instance.environment.in_task_instances[i]
        # in_task_instance.verbose = False
        if in_task_instance.task.task_type not in type_list:
            title = "{}".format(in_task_instance.task.task_type)
            plot_cost_to_go(in_task_instance.function, in_task_instance.environment.max_state,
                            plot_title=title, save_plot=True)
            type_list.append(in_task_instance.task.task_type)

    # evaluate root policy (between task policy)
    # testing_manager.experiment_instance.verbose = False
    type_list = []
    for i in range(testing_manager.experiment_instance.environment.num_tasks):
        in_task_instance = testing_manager.experiment_instance.environment.in_task_instances[i]
        if in_task_instance.task.task_type not in type_list:
            function = testing_manager.experiment_instance.function.functions[i]
            title = "{}-switch".format(in_task_instance.task.task_type)
            plot_cost_to_go(function, in_task_instance.environment.max_state, plot_title=title,
                            save_plot=True, is_switching=True)
            type_list.append(in_task_instance.task.task_type)

    total_reward_at_step, reward_at_step, costs_at_step, observation_sequence, task_at_step = \
        testing_manager.test_task_switching_policy()

    # adjust observation trajectory and calculate action sequence
    state_sequence = []
    index = 0
    for observation in observation_sequence:
        state_sequence.append(str(observation.task_id) + "," + str(observation.state_estimate))
        observation.time = float(index)
        observation.reward = reward_at_step[index]
        observation.cost = costs_at_step[index]
        index += 1

    print("avg. reward per step: {}\n".format(total_reward_at_step[-1] / len(total_reward_at_step)))
    for i in range(testing_manager.experiment_instance.environment.num_tasks):
        print("task {}: {}".format(i, colors[i]))

    max_val = np.max(reward_at_step)
    max_cost = np.max(costs_at_step)
    if max_cost > max_val:
        max_val = max_cost
    reward_at_step, state_sequence, task_at_step, costs_at_step = sparse_output(reward_at_step, state_sequence,
                                                                                task_at_step, costs_at_step)
    plot_policy_evaluation(reward_at_step, state_sequence, task_at_step=task_at_step, costs_at_step=costs_at_step,
                           max_value=max_val, plot_title='reward ' + str(total_reward_at_step[-1]))
    learning_manager.reset()


def add_trajectories_to_task(task_list, trajectory_file):
    with open(trajectory_file, 'rb') as f:
        trajectory_list = pickle.load(f)
    for i in range(len(task_list)):
        task_list[i].trajectories = trajectory_list[task_list[i].task_type]


def add_state_distribution_to_task(task_list, distribution_list):
    for i in range(len(task_list)):
        task_list[i].delta_state_distribution = distribution_list[task_list[i].task_type]


if __name__ == '__main__':
    main()
