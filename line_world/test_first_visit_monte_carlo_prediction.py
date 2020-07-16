from algorithms import *
from line_world import *
from policies import tabular_uniform_random_policy

if __name__ == "__main__":
    import time

    start_time = time.time()
    Pi = {}
    V = first_visit_monte_carlo_prediction(Pi, is_terminal, reset, step, get_possible_actions, 
    									   max_steps_per_episode=10, episodes_count=10000,
    									   action_dim = action_dim)

    print("--- %s seconds ---" % (time.time() - start_time))
    print(V)
