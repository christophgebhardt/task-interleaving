from LearnTaskSwitching import LearnTaskSwitching
from plot_utilities import plot_reward_learning, plot_policy, plot_in_task_state_value_functions, plot_high_level_state_value_functions
from utils import get_agent_state_value_functions, load_pkl_file, add_trajectories_to_task, add_state_distribution_to_task


def main():
    participant_name = "0B8VVB89"

    # files necessary to simulate the RL environment of the agent
    task_file = "data/environments/" + participant_name + "_train_env.pkl"
    trajectory_file = 'data/trajectories/' + participant_name + '_train.pkl'
    state_distribution_file = 'data/distributions/' + participant_name + '_state.pkl'
    # environment file for testing
    test_file = "data/environments/" + participant_name + "_test_env.pkl"

    # execution control
    verbose_switching = 1

    # learning parameters
    num_episodes = 250              # number of episodes
    learn_pursue_costs = True       # flag indicating that resumption costs are considered on the lower level
    learn_switch_costs = True       # flag indicating that resumption costs are considered on the higher level
    is_reward_from_obs = True       # flag indicating that reward comes from collected trajectories of participant
    is_smdp = True                  # flag indicating that underlying decision process is an SMDP

    # load pickle files
    task_list = load_pkl_file(task_file)
    distribution_list = load_pkl_file(state_distribution_file)
    trajectory_list = load_pkl_file(trajectory_file)

    add_state_distribution_to_task(task_list, distribution_list)
    add_trajectories_to_task(task_list, trajectory_list)

    learning_manager = LearnTaskSwitching(task_list, num_episodes=num_episodes, verbose_switching=verbose_switching,
                                          learn_pursue_costs=learn_pursue_costs, learn_switch_costs=learn_switch_costs,
                                          is_reward_from_obs=is_reward_from_obs)
    # learn task switching policy
    reward_per_episode, avg_reward_per_episode = learning_manager.learn_task_switching_policy()
    plot_reward_learning(reward_per_episode, expected_reward_array=avg_reward_per_episode, plot_title="learning curve")

    # get the tasks of the test environment
    test_task_list = load_pkl_file(test_file)
    add_trajectories_to_task(test_task_list, trajectory_list)
    add_state_distribution_to_task(test_task_list, distribution_list)

    # get state value functions form trained HRL agent
    learned_functions = get_agent_state_value_functions(learning_manager)
    # create the learning manager for the test environment
    testing_manager = LearnTaskSwitching(test_task_list, is_smdp=is_smdp, verbose_switching=verbose_switching,
                                         init_functions=learned_functions)

    plot_in_task_state_value_functions(testing_manager)

    plot_high_level_state_value_functions(testing_manager)

    plot_policy(testing_manager)


if __name__ == '__main__':
    main()
