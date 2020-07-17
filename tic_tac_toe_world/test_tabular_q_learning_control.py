from typing import Callable
from tic_tac_toe_world import *
from algorithms import *
#import matplotlib.pyplot as plt


if __name__ == "__main__":
    Q, Pi, learning_rates = tabular_q_learning_control(reset,
                                       is_terminal, step,
                                       get_possible_actions,
                                       epsilon=0.75,
                                       episodes_count = 50000,
                                       max_steps_per_episode=100,
                                       eval_results = True )
    print(Q)
    print(Pi)
    success_rate, failure_rate = eval(Pi)
    print( "Success rate: ", success_rate, "Failure rate: ", failure_rate)
    plt.plot(learning_rates)
    plt.show()


    
