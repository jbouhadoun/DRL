from typing import Callable
import numpy as np


def step_until_the_end_of_the_episode_and_return_history(
        s0: int,
        pi: np.ndarray,
        is_terminal_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        max_steps: int = 10,
        action_dim = 9,
) -> \
        ([int], [int], [int], [float]):
    s_list = []
    a_list = []
    s_p_list = []
    r_list = []
    st = s0
    
    
    steps_count = 0
    while not is_terminal_func(st) and steps_count < max_steps:
        actions = np.arange(action_dim)

        if st in pi.keys():
            at = np.random.choice(actions, p=pi[st])
        else:
            at = np.random.choice(get_possible_actions(st))

        st_p, rt_p, t, _ = step_func(at)
        s_list.append(st)
        a_list.append(at)
        s_p_list.append(st_p)
        r_list.append(rt_p)
        st = st_p
        steps_count += 1

    return s_list, a_list, s_p_list, r_list


def eval(Pi, reset, step, get_possible_actions, nb_episodes_test = 100, human = False, action_dim = 9, max_episode_steps=100):
    
    done = False
    state = reset()
    successes = 0
    fails = 0
    steps = 0
    rewards = 0
    for i in range(nb_episodes_test):
        state = reset()
        done = False
        t = 0
        while not done and t < max_episode_steps:
            t += 1
            if state in Pi.keys():
                action = np.random.choice(np.arange(action_dim), p=Pi[state])
            else:
                action = np.random.choice(get_possible_actions(state))

            state, reward, done, _ = step(action) 
            rewards += reward
            if reward == 1:
                successes +=1
            elif reward == -1:
                fails+=1
        steps += t


    return (successes*1.0/nb_episodes_test, fails*1.0/nb_episodes_test, 
            steps*1.0/nb_episodes_test, rewards*1.0/nb_episodes_test)


def softmax_gradient(s):
    s = s.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

"""def step_until_the_end_of_the_episode_and_return_history(
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
    return s_list, a_list, s_p_list, r_list"""







