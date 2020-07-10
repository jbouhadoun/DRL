from typing import Callable
import numpy as np


def step_until_the_end_of_the_episode_and_return_history(
        s0: int,
        pi: np.ndarray,
        is_terminal_func: Callable,
        step_func: Callable,
        max_steps: int = 10
) -> \
        ([int], [int], [int], [float]):
    s_list = []
    a_list = []
    s_p_list = []
    r_list = []
    st = s0
    actions = np.arange(pi.shape[1])
    steps_count = 0
    while not is_terminal_func(st) and steps_count < max_steps:
        at = np.random.choice(actions, p=pi[st])
        st_p, rt_p, t = step_func(st, at)
        s_list.append(st)
        a_list.append(at)
        s_p_list.append(st_p)
        r_list.append(rt_p)
        st = st_p
        steps_count += 1
    return s_list, a_list, s_p_list, r_list







def step_until_the_end_of_the_episode_and_return_history_v2(
        s0: int,
        pi: np.ndarray,
        is_terminal_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        max_steps: int = 10
) -> \
        ([int], [int], [int], [float]):
    s_list = []
    a_list = []
    s_p_list = []
    r_list = []
    st = s0
    action_dim = 9
    
    steps_count = 0
    while not is_terminal_func(st) and steps_count < max_steps:
        actions = np.arange(action_dim)

        if st in pi.keys():
            at = np.random.choice(actions, p=pi[st])
        else:
            at = np.random.choice(get_possible_actions(st))

        st_p, rt_p, t = step_func(at)
        s_list.append(st)
        a_list.append(at)
        s_p_list.append(st_p)
        r_list.append(rt_p)
        st = st_p
        steps_count += 1

    return s_list, a_list, s_p_list, r_list