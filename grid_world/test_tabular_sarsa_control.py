from algorithms import *
from grid_world import *

if __name__ == "__main__":
    Q, Pi = tabular_sarsa_control(reset,
                                  is_terminal, step,
                                  get_possible_actions,
                                  epsilon=0.2,
                                  max_steps_per_episode=100,
                                  action_dim = 4)
    print(Q)
    print(Pi)
