from tic_tac_toe_world import *
from typing import Callable
from utils import step_until_the_end_of_the_episode_and_return_history_v2
from algorithms import *



if __name__ == "__main__":
    Q, Pi = off_policy_monte_carlo_control_v2(reset,
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
    		state, reward, done, _ = step(action)
    		if reward == 1:
    			successes +=1
    		elif reward == -1:
    			fails+=1


    print("Success rate: ", successes*1.0/nb_episodes_test, "Failure rate: ", fails*1.0/nb_episodes_test)


