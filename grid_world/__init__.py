import numpy as np

width = 4
height = 4
num_states = width * height

S = np.arange(num_states)
A = np.arange(4)  # 0: left, 1: Right, 2: Up, 3: Down
T = np.array([width - 1, num_states - 1])
P = np.zeros((len(S), len(A), len(S), 2))

current_state = 0
action_dim = len(A)

for s in S:
    if (s % width) == 0:
        P[s, 0, s, 0] = 1.0
    else:
        P[s, 0, s - 1, 0] = 1.0
    if (s + 1) % width == 0:
        P[s, 1, s, 0] = 1.0
    else:
        P[s, 1, s + 1, 0] = 1.0
    if s < width:
        P[s, 2, s, 0] = 1.0
    else:
        P[s, 2, s - width, 0] = 1.0
    if s >= (num_states - width):
        P[s, 3, s, 0] = 1.0
    else:
        P[s, 3, s + width, 0] = 1.0

P[width - 1, :, :, 0] = 0.0
P[num_states - 1, :, :, 0] = 0.0

P[:, :, width - 1, 1] = -5.0
P[:, :, num_states - 1, 1] = 1.0


def reset() -> int:
    global current_state
    current_state = 0
    return current_state


def is_terminal(state: int) -> bool:
    return state in T

def step(state:int, a:int)-> (int, float, bool):
    assert (current_state not in T)
    s_p = np.random.choice(S, p=P[current_state, a, :, 0])
    r = P[current_state, a, s_p, 1]
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
