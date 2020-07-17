from algorithms import tabular_q_learning_control
from grid_world import *

if __name__ == "__main__":
    Q, Pi, learning_rates = tabular_q_learning_control(reset,
                                       is_terminal, step,
                                       get_possible_actions,
                                       epsilon=0.75,
                                       max_steps_per_episode=100,
                                       eval_results = True,
                                       action_dim = action_dim )
    print(Q)
    print(Pi)
