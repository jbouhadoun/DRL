from tic_tac_toe_world import *
from typing import Callable
from utils import step_until_the_end_of_the_episode_and_return_history_v2
from algorithms import *





if __name__ == "__main__":
    import time

    start_time = time.time()
    #Pi = tabular_uniform_random_policy(S.shape[0], A.shape[0])
    Pi = {}
    V = first_visit_monte_carlo_prediction_v2(Pi, is_terminal, reset, step, max_steps_per_episode=10, episodes_count=100000)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print(V)
    print(len(V.keys()))