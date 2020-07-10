from tic_tac_toe_world import *
from typing import Callable

def first_visit_monte_carlo_prediction(
        pi: dict,
        is_terminal_func: Callable,
        reset_func: Callable,
        step_func: Callable,
        episodes_count: int = 100000,
        max_steps_per_episode: int = 100,
        gamma: float = 0.99,
        exploring_start: bool = False
) -> np.ndarray:
    #states = np.arange(pi.shape[0])
    #V = np.random.random(pi.shape[0])

    pi = {}
    V = {}

    """for s in states:
        if is_terminal_func(s):
            V[s] = 0"""

    # returns = np.zeros(V.shape[0])
    # returns_count = np.zeros(V.shape[0])

    returns = {}
    returns_count = {}

    for episode_id in range(episodes_count):
        # s0 = np.random.choice(states) if exploring_start else reset_func()

        s0 =  reset_func()
        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_return_history(s0, pi, is_terminal_func,
                                                                                         step_func,
                                                                                         max_steps_per_episode)
        G = 0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            if st in s_list[0:t]:
                continue
            if st not in returns.keys():
            	returns[st] = 0
            	returns_count[st] = 0
            returns[st] += G
            returns_count[st] += 1
            V[st] = returns[st] / returns_count[st]
    return V

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
    
    steps_count = 0
    while not is_terminal_func(st) and steps_count < max_steps:
    	actions = get_possible_actions(st)
    	if st in pi.keys():
    		at = np.random.choice(actions, p=pi[st])
    	else:
    		at = np.random.choice(actions)

    	st_p, rt_p, t = step_func(at)
    	s_list.append(st)
    	a_list.append(at)
    	s_p_list.append(st_p)
    	r_list.append(rt_p)
    	st = st_p
    	steps_count += 1

    return s_list, a_list, s_p_list, r_list


if __name__ == "__main__":
    import time

    start_time = time.time()
    #Pi = tabular_uniform_random_policy(S.shape[0], A.shape[0])
    Pi = {}
    V = first_visit_monte_carlo_prediction(Pi, is_terminal, reset, step, max_steps_per_episode=10, episodes_count=100000)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print(V)
    print(len(V.keys()))