import numpy as np


def update_init_tables(learning_manager, init_tables=None):
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


def adjust_resumption_cost(task_list, const_cost, calc_scaler, check_scaler, read_scaler, type_scaler):
    for i in range(len(task_list)):
        # print(task_list[i].task_type)
        # print(task_list[i].resumption)
        if task_list[i].task_type == "calculating":
            task_list[i].resumption *= calc_scaler
        elif task_list[i].task_type == "checking":
            task_list[i].resumption *= check_scaler
        elif task_list[i].task_type == "reading":
            task_list[i].resumption *= read_scaler
        elif task_list[i].task_type == "typing":
            task_list[i].resumption *= type_scaler
        # print(task_list[i].resumption)
        task_list[i].resumption += const_cost
        # print(task_list[i].resumption)
        

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