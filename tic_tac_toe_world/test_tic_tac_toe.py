from tic_tac_toe_world import *
import random

if __name__ == "__main__":
    import time
    state = reset()
    render()

    done = False
    while not done:
    	action = random.choice(get_possible_actions(state))
    	state, reward, done = step(action)
    	render()