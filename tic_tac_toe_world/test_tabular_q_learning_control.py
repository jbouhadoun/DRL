from typing import Callable
from tic_tac_toe_world import *
import matplotlib.pyplot as plt

def tabular_q_learning_control(
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 50000,
        max_steps_per_episode: int = 10,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        gamma: float = 0.99,
) -> (np.ndarray, np.ndarray):
    # states = np.arange(states_count)
    # actions = np.arange(actions_count)
    action_dim = 9
    q = {} # np.random.random((states_count, actions_count))
    """for s in states:
        if is_terminal_func(s):
            q[s, :] = 0.0"""

    learning_rates = []
    for episode_id in range(episodes_count):
        s = reset_func()
        step = 0
        

        while not is_terminal_func(s) and step < max_steps_per_episode:

        	if s not in q.keys():
        		q[s]=np.random.random(action_dim)
	        possible_actions = get_possible_actions(s)

	        for action in range(action_dim):
	        	if action not in possible_actions:
	        		q[s][action] = -999999

	        rdm = np.random.random()
	        possible_actions = get_possible_actions(s)
	        a = np.random.choice(possible_actions) if rdm < epsilon else np.argmax(q[s])
	        (s_p, r, t) = step_func(a)
	        if s_p not in q.keys():
        		q[s_p]=np.random.random(action_dim)
	        possible_actions = get_possible_actions(s_p)

	        for action in range(action_dim):  
	        	if action not in possible_actions:
	        		q[s_p][action] = -999999
	        q[s][a] += alpha * (r + gamma * np.max(q[s_p]) - q[s][a])
	        s = s_p
	        step += 1
        pi = {}
        for s in q.keys():
            pi[s] = np.zeros(action_dim)
            pi[s][np.argmax(q[s])] = 1.0
        learning_rates.append(eval(pi)[0])

    

    return q, pi, learning_rates

def eval(Pi,nb_episodes_test = 100, human = False):
    done = False
    state = reset()
    successes = 0
    fails = 0
    action_dim = 9
    for i in range(nb_episodes_test):
        state = reset(human=human)
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


    return (successes*1.0/nb_episodes_test, fails*1.0/nb_episodes_test)


if __name__ == "__main__":
    Q, Pi, learning_rates = tabular_q_learning_control(reset,
                                       is_terminal, step,
                                       get_possible_actions,
                                       epsilon=0.75,
                                       max_steps_per_episode=100)
    print(Q)
    print(Pi)
    success_rate, failure_rate = eval(Pi)
    print( "Success rate: ", success_rate, "Failure rate: ", failure_rate)
    plt.plot(learning_rates)
    plt.show()


    
