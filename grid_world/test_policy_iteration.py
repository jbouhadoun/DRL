from algorithms import policy_iteration
from grid_world import *

if __name__ == "__main__":
    V, Pi = policy_iteration(S, A, P, T, reset, step, get_possible_actions, action_dim=action_dim)
    print(V)
    print(Pi)
