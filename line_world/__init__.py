from typing import Callable

import numpy as np

num_states = 50
S = np.arange(num_states)
A = np.array([0, 1])  # 0: left, 1 : right
T = np.array([0, num_states - 1])
P = np.zeros((len(S), len(A), len(S), 2))

action_dim = len(A)
current_state = num_states // 2

for s in S[1:-1]:
    P[s, 0, s - 1, 0] = 1.0
    P[s, 1, s + 1, 0] = 1.0
P[1, 0, 0, 1] = -1.0
P[num_states - 2, 1, num_states - 1, 1] = 1.0


def reset() -> int:
	global current_state
	current_state = num_states // 2
	return current_state


def is_terminal(state: int) -> bool:
    return state in T


def step(state: int, a: int) -> (int, float, bool):
    assert (state not in T)
    s_p = np.random.choice(S, p=P[state, a, :, 0])
    r = P[state, a, s_p, 1]
    return s_p, r, (s_p in T)

def step(a: int) -> (int, float, bool, dict):
    global current_state
    assert (current_state not in T)
    s_p = np.random.choice(S, p=P[current_state, a, :, 0])
    r = P[current_state, a, s_p, 1]
    current_state = s_p
    return s_p, r, (s_p in T), None

def get_possible_actions(state):
    return A

def set_current_state(state):
    global current_state
    current_state = state

def get_random_state():
    return np.random.choice(S)