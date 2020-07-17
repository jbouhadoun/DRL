from tic_tac_toe_world import *
from typing import Callable
from algorithms import *



if __name__ == "__main__":
	action_dim = 9
	Q, Pi = monte_carlo_with_exploring_starts_control(is_terminal, step, reset, get_possible_actions,
													  get_random_state, set_current_state, 
													  episodes_count=50000,
													  eval_results = True,
													  max_steps_per_episode=100)

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
			state, reward, done, _ = step(action)
			if reward == 1:
				successes +=1
			elif reward == -1:
				fails+=1

	print("Success rate: ", successes*1.0/nb_episodes_test, "Failure rate: ", fails*1.0/nb_episodes_test)




