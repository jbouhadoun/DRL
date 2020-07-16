import numpy as np


def tabular_uniform_random_policy(space_size: int, action_size: int):
    return np.ones((space_size, action_size)) / action_size


def softmax_policy(weights, state):
	prod = state.dot(weights)
	return np.exp(prod)/np.sum(np.exp(prod))

