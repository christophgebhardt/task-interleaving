from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
    sns.set()
except:
    print("seaborn not available")
font_size = 14


def get_reward_cost_figure(task_len, reward_at_step, cost_at_step, reward, cost, **kwargs):
    max_value = -1
    reward_values = -1
    reward_time = -1
    cost_values = -1
    cost_time = -1
    steps = -1
    save_plot = False
    task_color = -1
    task_num = 0
    plot_title = None
    return_figure = False
    for key, value in kwargs.items():
        if key == "max_value":
            max_value = value
        elif key == "reward_values":
            reward_values = value
        elif key == "reward_time":
            reward_time = value
        elif key == "cost_values":
            cost_values = value
        elif key == "cost_time":
            cost_time = value
        elif key == "steps":
            steps = value
        elif key == "save_plot":
            save_plot = value
        elif key == "task_num":
            task_num = value
        elif key == "task_color":
            task_color = value
        elif key == "plot_title":
            plot_title = value
        elif key == "return_figure":
            return_figure = value

    if not isinstance(steps, np.ndarray):
        steps = np.arange(0, len(reward)) * 0.1

    fig = plt.figure()
    if plot_title is not None:
        plt.suptitle(plot_title)
    ax = fig.add_subplot(121)
    # ax = fig.add_subplot(221)
    ax.plot(steps, reward, 'g')
    if isinstance(reward_values, np.ndarray) and isinstance(reward_time, np.ndarray):
        ax.scatter(reward_time, reward_values, c="green")
    # start costs
    if not isinstance(cost, np.ndarray):
        cost = np.ones(len(steps)) * cost
    ax.plot(steps, cost, 'r')
    if isinstance(cost_values, np.ndarray) and isinstance(cost_time, np.ndarray):
        ax.scatter(cost_time, cost_values, c="red")
    ax.set_title("Reward & Cost")
    # end costs
    # ax.set_title("Reward")
    if task_len != -1:
        ax.set_xlim([0, task_len])
    # if max_value != -1:
    #     ax.set_ylim([0, max_value])

    # ax = fig.add_subplot(223)
    # if not isinstance(cost, np.ndarray):
    #     cost = np.ones(len(steps)) * cost
    # ax.plot(steps, cost, 'r')
    # if isinstance(cost_values, np.ndarray) and isinstance(cost_time, np.ndarray):
    #     ax.scatter(cost_time, cost_values, c="red")
    # ax.set_title("Cost")
    # if task_len != -1:
    #     ax.set_xlim([0, task_len])
    # if max_value != -1:
    #     ax.set_ylim([0, max_value])

    ax1 = fig.add_subplot(222)
    x = np.arange(task_len)
    if task_color != -1:
        ax1.bar(x, reward_at_step, color=task_color)
    else:
        ax1.bar(x, reward_at_step)
    # ax1.set_ylim([0, 5])  # limits for y-axis
    x_ticks = [0]
    for i in range(5, len(x), 5):
        x_ticks.append(int(x[i]))
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks, fontsize=font_size)
    # ax1.set_xticklabels(x_ticklabels)
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(x.astype(int), fontsize=font_size)
    if max_value != -1:
        ax1.set_ylim([0, max_value])
        y_ticks = [0, 5, 10]
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_ticks, fontsize=font_size)

    ax1.set_title("Discrete reward")

    ax = fig.add_subplot(224, sharex=ax1)
    x = np.arange(task_len)
    if not isinstance(cost_at_step, np.ndarray):
        cost_at_step = np.ones(task_len) * cost_at_step
    cost_at_step = cost_at_step * -1
    ax.bar(x, cost_at_step, color=task_color)
    # ax.set_ylim([0, 5])  # limits for y-axis
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=font_size)
    # ax.set_xticks(x)
    # ax.set_xticklabels(x.astype(int), fontsize=font_size)
    if max_value != -1:
        ax.set_ylim([-max_value, 0])
        y_ticks = [-int(max_value), -5, 0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, fontsize=font_size)
    # ax.set_title("Discrete cost")

    plt.subplots_adjust(left=0.05, right=0.95)
    f = plt.gcf()
    default_size = f.get_size_inches()
    f.set_size_inches((default_size[0] * 1.5, default_size[1] * 1))
    if save_plot:
        if plot_title is None:
            plt.savefig('task' + str(task_num) + '.png')  # save graph to folder
        else:
            plt.savefig('./plots/' + plot_title + '.png')
        plt.close()
    elif return_figure:
        return fig
    else:
        plt.show()


def plot_reward_learning(reward_per_run_array, **kwargs):
    save_plot = False
    expected_reward_array = -1
    return_figure = False
    plot_title = -1
    for key, value in kwargs.items():
        if key == "save_plot":
            save_plot = value
        elif key == "expected_reward_array":
            expected_reward_array = value
        elif key == "return_figure":
            return_figure = value
        elif key == "plot_title":
            plot_title = value
    fig = plt.figure()
    plt.plot(reward_per_run_array, label="Reward at run", color='orange')
    if isinstance(expected_reward_array, np.ndarray):
        plt.plot(expected_reward_array, label="Avg. reward", color='green')
    plt.draw()
    # plt.ylim(-0.5, 1)  # limits for the y-axis
    plt.legend(loc='best')  # display the graph key
    plt.subplots_adjust(left=0.05, right=0.95)
    if save_plot:
        if plot_title == -1:
            plt.savefig('reward_graph.png')
        else:
            plt.savefig('./plots/' + plot_title + '.png')
        plt.close()
    elif return_figure:
        return fig
    else:
        plt.show()


def plot_policy_evaluation(reward_at_step, action_sequence, **kwargs):
    save_plot = False
    task_at_step = -1
    plot_title = -1
    costs_at_step = -1
    return_figure = False
    max_value = -1
    num_test_states = None
    for key, value in kwargs.items():
        if key == "save_plot":
            save_plot = value
        elif key == "task_at_step":
            task_at_step = value
        elif key == "plot_title":
            plot_title = value
        elif key == "costs_at_step":
            costs_at_step = -1 * value
        elif key == "return_figure":
            return_figure = value
        elif key == "max_value":
            max_value = value
        if key == "num_test_states":
            num_test_states = value

    if action_sequence.dtype.type is not np.str_:
        action_sequence = action_sequence.astype(int)

    fig = plt.figure()
    if plot_title != -1:
        plt.suptitle(plot_title)

    if isinstance(costs_at_step, np.ndarray):
        # ax = fig.add_subplot(312)
        ax1 = fig.add_subplot(211)
    else:
        ax1 = fig.add_subplot(212)

    reward_at_step[reward_at_step < 0.2] = 0.2
    x = np.arange(len(reward_at_step))
    if task_at_step != -1:
        ax1.bar(x, reward_at_step, color=task_at_step)
    else:
        ax1.bar(x, reward_at_step)
    # ax1.set_ylim([-0.1, 5])  # limits for y-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(action_sequence, fontsize=font_size)
    if max_value != -1:
        ax1.set_ylim([0, max_value])
    if num_test_states is not None:
        plt.axvline(x=num_test_states)

    if isinstance(costs_at_step, np.ndarray):
        costs_at_step[costs_at_step > -0.1] = -0.1
        x = np.arange(len(costs_at_step))
        # ax = fig.add_subplot(313)
        ax = fig.add_subplot(212, sharex=ax1)
        if task_at_step != -1:
            ax.bar(x, costs_at_step, color=task_at_step)
        else:
            ax.bar(x, costs_at_step)
        ax.set_xticks(x)
        ax.set_xticklabels(action_sequence, fontsize=font_size)
        if max_value != -1:
            ax.set_ylim([-max_value, 0])
        if num_test_states is not None:
            plt.axvline(x=num_test_states)

    plt.subplots_adjust(left=0.05, right=0.95)
    f = plt.gcf()
    default_size = f.get_size_inches()
    f.set_size_inches((default_size[0] * 2.5, default_size[1] * 1))
    if save_plot:
        if plot_title == -1:
            plt.savefig('./plots/policy.png')  # save graph to folder
        else:
            plt.savefig('./plots/' + plot_title + '.png')
        plt.close()
    elif return_figure:
        return fig
    else:
        plt.show()


def get_policy_evaluation_subplot(ax1, reward_at_step, action_sequence, **kwargs):
    task_at_step = -1
    costs_at_step = -1
    ax = None
    max_value = -1
    plot_title = None
    for key, value in kwargs.items():
        if key == "task_at_step":
            task_at_step = value
        elif key == "costs_at_step":
            costs_at_step = -1 * value
        elif key == "cost_ax":
            ax = value
        elif key == "max_value":
            max_value = value
        if key == "plot_title":
            plot_title = value

    if action_sequence.dtype.type is not np.str_:
        action_sequence = action_sequence.astype(int)

    reward_at_step[reward_at_step < 0.2] = 0.2
    x = np.arange(len(reward_at_step))
    if task_at_step != -1:
        ax1.bar(x, reward_at_step, color=task_at_step)
    else:
        ax1.bar(x, reward_at_step)
    # ax1.set_ylim([-0.1, 5])  # limits for y-axis
    ax1.set_xticks(x)
    # ax1.set_xticklabels(action_sequence, fontsize=font_size)
    ax1.set_xticklabels([])
    if max_value != -1:
        ax1.set_ylim([0, max_value])
    if plot_title is not None:
        ax1.set_ylabel(plot_title)

    if isinstance(costs_at_step, np.ndarray):
        costs_at_step[costs_at_step > -0.1] = -0.1
        x = np.arange(len(costs_at_step))
        if task_at_step != -1:
            ax.bar(x, costs_at_step, color=task_at_step)
        else:
            ax.bar(x, costs_at_step)
        ax.set_xticks(x)
        # ax.set_xticklabels(action_sequence, fontsize=font_size)
        ax.set_xticklabels([])
        if max_value != -1:
            ax.set_ylim([-max_value, 0])


def plot_cost_to_go(function, max_state, **kwargs):
    save_plot = False
    plot_title = -1
    return_figure = False
    is_switching = False
    for key, value in kwargs.items():
        if key == "save_plot":
            save_plot = value
        elif key == "plot_title":
            plot_title = value
        elif key == "return_figure":
            return_figure = value
        elif key == "is_switching":
            is_switching = value

    step_size = 1
    if ("TaskPursuing" in str(function) and "Function" in str(function.function)) or "LinearFunction" in str(function):
        step_size = 0.1

    # print("{} {} {}".format(plot_title, max_state, step_size))
    scroll_values = np.arange(0, max_state + step_size, step_size)
    values_0 = np.zeros(scroll_values.shape[0])
    values_1 = np.zeros(scroll_values.shape[0])
    # actions = np.zeros(scroll_values.shape[0])
    if "TaskPursuing" in str(function):
        function.expected_out_of_task_reward = 0
    for i in range(len(scroll_values)):
        vector = np.array([scroll_values[i]])
        val_0 = 0
        if not isinstance(function, list):
            val_0 = function.get_value(vector, 0)
        values_0[i] = val_0
        if not is_switching:
            val_1 = function.get_value(vector, 1)
            values_1[i] = val_1
        # action = 0
        # val = val_0
        # if not is_switching:
        #     if val_1 > val_0:
        #         action = 1
        #         val = val_1
        # actions[i] = action

    font_size = 35
    fig = plt.figure()
    plt.rcParams.update({'font.size': font_size})
    # if plot_title != -1:
    #     plt.suptitle(plot_title)
    # Plot the values for action 0.
    ax = fig.add_subplot(121)
    ax.plot(scroll_values, values_0)
    # ax.set_title("Action 0")
    # if step_size == 1:
    #     ax.set_xticks(scroll_values)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    # Plot the values for action 1.
    if not is_switching:
        ax = fig.add_subplot(122)
        ax.plot(scroll_values, values_1)
        # ax.set_title("Action 1")
        # if step_size == 1:
        #     ax.set_xticks(scroll_values)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

    f = plt.gcf()
    default_size = f.get_size_inches()
    f.set_size_inches((default_size[0] * 2.5, default_size[1] * 1))
    if save_plot:
        if plot_title == -1:
            plt.savefig('./plots/policy.png')  # save graph to folder
        else:
            plt.savefig('./plots/' + plot_title + '.png')
        plt.close()
    elif return_figure:
        return fig
    else:
        plt.show()


def plot_cost_to_go_2d(function, max_state, task_type, **kwargs):
    save_plot = False
    plot_title = -1
    return_figure = False
    is_switching = False
    for key, value in kwargs.items():
        if key == "save_plot":
            save_plot = value
        elif key == "plot_title":
            plot_title = value
        elif key == "return_figure":
            return_figure = value
        elif key == "is_switching":
            is_switching = value

    # print("{} {}".format(plot_title, max_state))
    scroll_values = np.arange(0, max_state + 0.1, 0.1)
    max_reward = get_max_reward(task_type)
    reward_values = np.arange(0, max_reward + 1)
    values_0 = np.zeros((scroll_values.shape[0], reward_values.shape[0]))
    values_1 = np.zeros((scroll_values.shape[0], reward_values.shape[0]))
    for i in range(len(scroll_values)):
        for j in range(len(reward_values)):
            vector = np.array([scroll_values[i], reward_values[j]])
            val_0 = function.get_value(vector, 0)
            values_0[i, j] = val_0
            if not is_switching:
                val_1 = function.get_value(vector, 1)
                values_1[i, j] = val_1

    scroll_values, reward_values = np.meshgrid(reward_values, scroll_values)
    fig = plt.figure()
    if plot_title != -1:
        plt.suptitle(plot_title)
    # Plot the values for action 0.
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(scroll_values, reward_values, values_0)
    ax.set_title("Action 0")
    ax.set_xlabel("Scroll")
    ax.set_ylabel("Reward")

    # Plot the values for action 1.
    if not is_switching:
        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(scroll_values, reward_values, values_1)
        ax.set_title("Action 1")
        ax.set_xlabel("Scroll")
        ax.set_ylabel("Reward")

    f = plt.gcf()
    default_size = f.get_size_inches()
    f.set_size_inches((default_size[0] * 2.5, default_size[1] * 1))
    if save_plot:
        if plot_title == -1:
            plt.savefig('./plots/policy.png')  # save graph to folder
        else:
            plt.savefig('./plots/' + plot_title + '.png')
        plt.close()
    elif return_figure:
        return fig
    else:
        plt.show()


def get_max_reward(task_type):
    if task_type == "reading":
        return 20
    elif task_type == "checking":
        return 16
    else:  # calculating or typing
        return 6