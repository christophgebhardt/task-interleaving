import numpy as np
import pickle


def get_agent_state_value_functions(learning_manager, init_tables=None):
    # copy learned tables into init tables for initializing test manager and later learning managers
    if init_tables is None:
        init_tables = [{}, {}]

    for type, func_dict in learning_manager.experiment_instance.function.function_dict.items():
        if type not in init_tables[0]:
            init_tables[0][type] = {}
        if isinstance(func_dict, dict):
            for type2, function in func_dict.items():
                if type2 not in init_tables[0][type]:
                    init_tables[0][type][type2] = function
        else:
            init_tables[0][type] = func_dict

    for instance in learning_manager.in_task_instances:
        if instance.task.task_type not in init_tables[1]:
            init_tables[1][instance.task.task_type] = instance.function.function

    return init_tables


def sparse_output(reward_at_step, state_sequence, task_at_step, costs_at_step):
    sp_reward_at_step = [0]
    sp_state_sequence = [state_sequence[0]]
    sp_task_at_step = [task_at_step[0]]
    sp_costs_at_step = [costs_at_step[0]]
    reward = reward_at_step[0]

    for i in range(1, len(state_sequence)):
        if state_sequence[i] != sp_state_sequence[-1]:
            sp_reward_at_step[-1] = reward
            sp_reward_at_step.append(0)
            sp_state_sequence.append(state_sequence[i])
            sp_task_at_step.append(task_at_step[i])
            sp_costs_at_step.append(costs_at_step[i])
            reward = 0

        reward += reward_at_step[i]

    return np.array(sp_reward_at_step), np.array(sp_state_sequence), sp_task_at_step, np.array(sp_costs_at_step)


def get_state_trajectory(trajectory):
    obs = trajectory[0]
    num_visited_states = 1
    start_time = obs.time
    obs.time = 0
    state_trajectory = [obs]
    for i in range(1, len(trajectory)):
        if trajectory[i].state_estimate != obs.state_estimate or trajectory[i].task_id != obs.task_id:
            num_visited_states += 1
            obs = trajectory[i]
            obs.time -= start_time
            state_trajectory.append(obs)

    return state_trajectory, num_visited_states


def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        pkl_obj = pickle.load(f)
    return pkl_obj


def add_trajectories_to_task(task_list, trajectory_list):
    for i in range(len(task_list)):
        task_list[i].trajectories = trajectory_list[task_list[i].task_type]


def add_state_distribution_to_task(task_list, distribution_list):
    for i in range(len(task_list)):
        task_list[i].delta_state_distribution = distribution_list[task_list[i].task_type]