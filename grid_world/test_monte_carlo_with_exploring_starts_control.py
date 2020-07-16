from algorithms import monte_carlo_with_exploring_starts_control
from grid_world import *

if __name__ == "__main__":

	Q, Pi = monte_carlo_with_exploring_starts_control(is_terminal, step, reset, get_possible_actions, 
												 get_random_state, set_current_state, 
												 episodes_count=10000, max_steps_per_episode=100, action_dim = action_dim)

	print(Q)
	print(Pi)
