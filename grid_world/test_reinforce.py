import numpy as np

from algorithms import reinforce
from grid_world import *

def reset_():
	return np.array([reset()])

def step_(a):
	s, a, r, d = step(a)
	s = np.array([s])
	return s, a, r, d

reinforce(reset_, step_, get_possible_actions, action_dim = 4)
