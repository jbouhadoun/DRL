from algorithms import *
from line_world import *

if __name__ == "__main__":
    Q, Pi = tabular_expected_sarsa_control(
                                  reset,
                                  is_terminal, step,
                                  get_possible_actions,
                                  epsilon=0.75,
                                  max_steps_per_episode=100,
                                  action_dim=action_dim)
    print(Q)
    print(Pi)
