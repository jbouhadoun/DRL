from algorithms import on_policy_first_visit_monte_carlo_control
from grid_world import *

if __name__ == "__main__":
    Q, Pi = on_policy_first_visit_monte_carlo_control(reset,
                                                      is_terminal, step,
                                                      get_possible_actions,
                                                      episodes_count=1000,
                                                      max_steps_per_episode=100,
                                                      action_dim = action_dim)
    print(Q)
    print(Pi)
    print(Pi.keys())
