from typing import Callable
from tic_tac_toe_world import *

def dyna_q_control(
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 10000,
        max_steps_per_episode: int = 10,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        gamma: float = 0.99,
        n: int = 2,
) -> (np.ndarray, np.ndarray):
    q = {} 
    model = {}

    action_dim = 9

    for episode_id in range(episodes_count):
        s = reset_func()
        if s not in q.keys():
        	q[s]=np.random.random(action_dim)

        possible_actions = get_possible_actions(s)

        for action in range(action_dim):
        	if action not in possible_actions:
        		q[s][action] = -99999

        
        step = 0
        while not is_terminal_func(s) and step < max_steps_per_episode:

            rdm = np.random.random()
            possible_actions = get_possible_actions(s)
            a = np.random.choice(possible_actions) if rdm < epsilon else np.argmax(q[s])
            s_p, r, done = step_func(a)
            
            model[(s, a)] = (s_p, r)



            if s_p not in q.keys():
                q[s_p] = np.random.random(action_dim)
                possible_actions = get_possible_actions(s_p)

                for action in range(action_dim):
                    if action not in possible_actions:
                        q[s_p][action] = -99999

                    if is_terminal_func(s_p):
                        q[s_p][action] = 0


            delta = r + gamma * q[s_p].max() - q[s][a]
            q[s][a] += alpha * delta
            s = s_p
            step += 1

            # perform n steps
            for _ in range(n):
                (s_,a_), (s_p, r) = random.choice(list(model.items()))

                # update
                delta = r + gamma * q[s_p].max() - q[s_][a_]
                q[s_][a_] += alpha * delta



    pi = {} # np.zeros_like(q)
    for s in q.keys():
    	possible_actions = get_possible_actions(s)
    	if not is_terminal_func(s):
	    	pi[s] = np.zeros(action_dim) 

	    	pi[s][np.argmax(q[s])] = 1.0 

    return q, pi





if __name__ == "__main__":
    Q, Pi = dyna_q_control(reset,
                                  is_terminal, step,
                                  get_possible_actions,
                                  epsilon=0.2,
                                  max_steps_per_episode=10)
    

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
