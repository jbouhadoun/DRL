from tic_tac_toe_world import *
from typing import Callable
from utils import step_until_the_end_of_the_episode_and_return_history_v2



def monte_carlo_with_exploring_starts_control(
        is_terminal_func: Callable,
        step_func: Callable,
        reset_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 1000000,
        max_steps_per_episode: int = 10,
        gamma: float = 0.99,
) -> (np.ndarray, np.ndarray):

    # states = np.arange(states_count)
    # actions = np.arange(actions_count)
    # pi = tabular_uniform_random_policy(states_count, actions_count)
    pi = {}
    # q = np.random.random((states_count, actions_count))
    q = {}
    action_dim = 9
    """for s in states:
        if is_terminal_func(s):
            q[s, :] = 0.0
            pi[s, :] = 0.0"""

    returns = {} #np.zeros((states_count, actions_count))
    returns_count = {} # np.zeros((states_count, actions_count))

    for episode_id in range(episodes_count):
        s0 = reset_func() # np.random.choice(states)

        if is_terminal_func(s0):
            continue
        actions = get_possible_actions(s0)
        a0 = np.random.choice(actions)
        s1, r1, t1 = step_func(a0)

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_return_history_v2(s1, pi, is_terminal_func,
                                                                                         step_func, get_possible_actions,
                                                                                         max_steps_per_episode)
        s_list = [s0] + s_list
        a_list = [a0] + a_list
        r_list = [r1] + r_list

        G = 0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]

            if (st, at) in zip(s_list[0:t], a_list[0:t]):
                continue
            
            possible_actions = get_possible_actions(st)
            if st not in returns.keys():
            	returns[st] = np.zeros(action_dim)
            	returns_count[st] = np.zeros(action_dim)
            	q[st] = np.zeros(action_dim)
            	pi[st]= np.zeros(action_dim)

            	for a in range(action_dim):
	            	if a not in possible_actions:
	            		q[st][a] = -999999

            returns[st][at] += G
            returns_count[st][at] += 1
            q[st][at] = returns[st][at] / returns_count[st][at]
            pi[st]= np.zeros(action_dim)
            
            pi[st][np.argmax(q[st])] = 1.0
    return q, pi

if __name__ == "__main__":
	action_dim = 9
	Q, Pi = monte_carlo_with_exploring_starts_control(is_terminal, step, reset, get_possible_actions, episodes_count=10000, max_steps_per_episode=100)
	# print(Q)
	# print(Pi)
	done = False
	state = reset()

	nb_episodes_test = 1000
	successes = 0
	fails = 0
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




