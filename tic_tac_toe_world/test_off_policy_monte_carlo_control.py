from tic_tac_toe_world import *
from typing import Callable
from utils import step_until_the_end_of_the_episode_and_return_history_v2

def off_policy_monte_carlo_control(
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 10000,
        max_steps_per_episode: int = 10,
        epsilon: float = 0.2,
        gamma: float = 0.99,
) -> (np.ndarray, np.ndarray):
    """states = np.arange(states_count)
    
    b = tabular_uniform_random_policy(states_count, actions_count)
    pi = np.zeros((states_count, actions_count))
    C = np.zeros((states_count, actions_count))
    q = np.random.random((states_count, actions_count))
    for s in states:
        if is_terminal_func(s):
            q[s, :] = 0.0
            pi[s, :] = 0.0
        pi[s, :] = 0.0
        pi[s, np.argmax(q[s, :])] = 1.0"""


    b = {}
    pi = {}
    C = {}
    q = {}
    action_dim = 9

    for episode_id in range(episodes_count):
        s0 = reset_func()

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_return_history_v2(s0, b, is_terminal_func,
                                                                                         step_func, get_possible_actions,
                                                                                         max_steps_per_episode)

        G = 0
        W = 1
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]
            possible_actions = get_possible_actions(st)
            if st not in C.keys():
            	C[st] = np.zeros(action_dim)
            	q[st] = np.zeros(action_dim)
            	pi[st] = np.zeros(action_dim)
            	b[st]=np.ones(action_dim)*1.0/len(possible_actions)
            	for a in range(action_dim):
	            	if a not in possible_actions:
	            		q[st][a] = -9999999
	            		b[st][a] = 0
            C[st][at] += W

            q[st][at] += W / C[st][at] * (G - q[st][at])
            pi[st] = np.zeros(action_dim)
            pi[st][np.argmax(q[st])] = 1.0

            if at != np.argmax(q[st]):
                break

            W = W / b[st][at]

    return q, pi

if __name__ == "__main__":
    Q, Pi = off_policy_monte_carlo_control(reset,
                                           is_terminal, step, get_possible_actions,
                                           episodes_count=100000,
                                           max_steps_per_episode=10)
    # print(Q)
    # print(Pi)

    done = False
    state = reset()
    nb_episodes_test = 1000
    successes = 0
    fails = 0
    action_dim = 9
    for i in range(nb_episodes_test):
    	state = reset()
    	done = False
    	while not done:
    		if state in Pi.keys():
    			action = np.random.choice(np.arange(action_dim), p=Pi[state])
    		else:
    			action = np.random.choice(get_possible_actions(state))
    		state, reward, done = step(action)
    		if reward == 1:
    			successes +=1
    		elif reward == -1:
    			fails+=1


    print("Success rate: ", successes*1.0/nb_episodes_test, "Failure rate: ", fails*1.0/nb_episodes_test)


