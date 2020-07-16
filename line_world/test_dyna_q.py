from typing import Callable
from line_world import *
from algorithms import *






if __name__ == "__main__":
    Q, Pi = dyna_q_control(reset,
                                  is_terminal, step,
                                  get_possible_actions,
                                  epsilon=0.2,
                                  max_steps_per_episode=100,
                                  action_dim= 2)
