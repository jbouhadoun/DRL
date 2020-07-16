from typing import Callable
from grid_world import *
from algorithms import *






if __name__ == "__main__":
    Q, Pi = dyna_q_control(reset,
                                  is_terminal, step,
                                  get_possible_actions,
                                  epsilon=0.2,
                                  max_steps_per_episode=100,
                                  action_dim= 4)


    print("Success rate: ", successes*1.0/nb_episodes_test, "Failure rate: ", fails*1.0/nb_episodes_test)
