from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import random
import os

from policies import *
from utils import *


def iterative_policy_evaluation(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        Pi: np.ndarray,
        gamma: float = 0.99,
        theta: float = 0.000001,
        V: np.ndarray = None
) -> np.ndarray:
    assert 0 <= gamma <= 1
    assert theta > 0

    if V is None:
        V = np.random.random((S.shape[0],))
        V[T] = 0.0
    while True:
        delta = 0
        for s in S:
            v_temp = V[s]
            tmp_sum = 0
            for a in A:
                for s_p in S:
                    tmp_sum += Pi[s, a] * P[s, a, s_p, 0] * (
                            P[s, a, s_p, 1] + gamma * V[s_p]
                    )
            V[s] = tmp_sum
            delta = np.maximum(delta, np.abs(tmp_sum - v_temp))
        if delta < theta:
            break
    return V


def policy_iteration(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        reset_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        gamma: float = 0.99,
        theta: float = 0.000001,
        eval_results: bool = True,
        action_dim: int = 2,
) -> (np.ndarray, np.ndarray):
    Pi = tabular_uniform_random_policy(S.shape[0], A.shape[0])
    V = np.random.random((S.shape[0],))
    V[T] = 0.0

    print("Value: ")
    print(V)

    print("Policy: ")
    print(Pi)

    pi = {}
    if eval_results:
    	pi = {}
    	directory = 'Data/'
    	if not os.path.exists(directory):
    		os.makedirs(directory)
    	results_file = open(directory+'policy_iteration_results.txt', "a") 
    	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
    	results_file.write(f"0 {successes} {fails} {steps} {rewards}\n")
    e=0
    while True:
        # Policy evalutation
        V = iterative_policy_evaluation(S, A, P, T, Pi, gamma, theta, V)
        
        e += 1

        # Policy impovement
        policy_stable = True

        for s in S:

            old_action = np.argmax(Pi[s])
            best_action = 0
            best_action_score = -9999999999999
            for a in A:
                tmp_sum = 0
                for s_p in S:
                    tmp_sum += P[s, a, s_p, 0] * (
                            P[s, a, s_p, 1] + gamma * V[s_p]
                    )
                if tmp_sum > best_action_score:
                    best_action = a
                    best_action_score = tmp_sum
            Pi[s] = 0.0
            pi[s] = np.zeros(action_dim)
            pi[s][best_action] = 1.0
            Pi[s, best_action] = 1.0
            if best_action != old_action:
                policy_stable = False
        if eval_results:
            	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
            	results_file.write(f"{e} {successes} {fails} {steps} {rewards}\n")

        if policy_stable:
            break
        print("Value: ")
        print(V)

        print("Policy: ")
        print(Pi)

    if eval_results:
        results_file.close()

    return V, Pi


def value_iteration(
        S: np.ndarray,
        A: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        reset_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        gamma: float = 0.99,
        theta: float = 0.000001,
        eval_results: bool = True,
        action_dim: int = 2,
) -> (np.ndarray, np.ndarray):
    assert 0 <= gamma <= 1
    assert theta > 0
    if eval_results:
    	pi = {}
    	directory = 'Data/'
    	if not os.path.exists(directory):
    		os.makedirs(directory)
    	results_file = open(directory+'value_iteration_results.txt', "a") 
    	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
    	results_file.write(f"0 {successes} {fails} {steps} {rewards}\n")

    V = np.random.random((S.shape[0],))
    V[T] = 0.0
    e = 0
    while True:
        delta = 0
        e+=1
        for s in S:
            v_temp = V[s]
            best_score = -9999999999
            for a in A:
                tmp_sum = 0
                for s_p in S:
                    tmp_sum += P[s, a, s_p, 0] * (
                            P[s, a, s_p, 1] + gamma * V[s_p]
                    )
                if best_score < tmp_sum:
                    best_score = tmp_sum
            V[s] = best_score
            delta = np.maximum(delta, np.abs(V[s] - v_temp))
        if delta < theta:
            break
        if eval_results:
        	pi = {}
        	Pi = np.zeros((S.shape[0], A.shape[0]))
        	for s in S:
		        best_action = 0
		        best_action_score = -9999999999999
		        for a in A:
		            tmp_sum = 0
		            for s_p in S:
		                tmp_sum += P[s, a, s_p, 0] * (
		                        P[s, a, s_p, 1] + gamma * V[s_p]
		                )
		            if tmp_sum > best_action_score:
		                best_action = a
		                best_action_score = tmp_sum
		        Pi[s] = 0.0
		        Pi[s, best_action] = 1.0
		        pi[s] = np.zeros(action_dim)
		        pi[s][best_action]=1.0

	        successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
	        results_file.write(f"{e} {successes} {fails} {steps} {rewards}\n")



    Pi = np.zeros((S.shape[0], A.shape[0]))
    for s in S:
        best_action = 0
        best_action_score = -9999999999999
        for a in A:
            tmp_sum = 0
            for s_p in S:
                tmp_sum += P[s, a, s_p, 0] * (
                        P[s, a, s_p, 1] + gamma * V[s_p]
                )
            if tmp_sum > best_action_score:
                best_action = a
                best_action_score = tmp_sum
        Pi[s] = 0.0
        Pi[s, best_action] = 1.0
    if eval_results:
    	results_file.close()
    return V, Pi


def first_visit_monte_carlo_prediction(
        pi: dict,
        is_terminal_func: Callable,
        reset_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 1000,
        max_steps_per_episode: int = 100,
        gamma: float = 0.99,
        exploring_start: bool = False,
        action_dim: int = 9,
) -> np.ndarray:

    pi = {}
    V = {}

    returns = {}
    returns_count = {}

    for episode_id in range(episodes_count):
        # s0 = np.random.choice(states) if exploring_start else reset_func()

        s0 =  reset_func()
        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_return_history(s0, pi, is_terminal_func,
                                                                                         step_func, get_possible_actions,
                                                                                         max_steps_per_episode, action_dim = action_dim)
        G = 0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            if st in s_list[0:t]:
                continue
            if st not in returns.keys():
                returns[st] = 0
                returns_count[st] = 0
            returns[st] += G
            returns_count[st] += 1
            V[st] = returns[st] / returns_count[st]
    return V


def monte_carlo_with_exploring_starts_control(
        is_terminal_func: Callable,
        step_func: Callable,
        reset_func: Callable,
        get_possible_actions: Callable,
        get_random_state: Callable,
        set_current_state: Callable,
        episodes_count: int = 1000,
        max_steps_per_episode: int = 100,
        eval_results = True,
        gamma: float = 0.99,
        action_dim: int = 9,
) -> (np.ndarray, np.ndarray):

 
    pi = {}
    q = {}
    if eval_results:
    	directory = 'Data/'
    	if not os.path.exists(directory):
    		os.makedirs(directory)
    	results_file = open(directory+'monte_carlo_with_exploring_starts_control_results.txt', "a") 
    	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
    	results_file.write(f"0 {successes} {fails} {steps} {rewards}\n")

    returns = {}
    returns_count = {} 

    for episode_id in range(episodes_count):
        s0 = get_random_state() 
        set_current_state(s0)
        #s0 = reset_func()


        if is_terminal_func(s0):
            continue
        actions = get_possible_actions(s0)
        a0 = np.random.choice(actions)
        s1, r1, t1, _ = step_func(a0)

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_return_history(s1, pi, is_terminal_func,
                                                                                         step_func, get_possible_actions,
                                                                                         max_steps_per_episode, action_dim = action_dim)
        s_list = [s0] + s_list
        a_list = [a0] + a_list
        r_list = [r1] + r_list

        G = 0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]

            if (st, at) in zip(s_list[0:t], a_list[0:t]):
                continue
            
            possible_actions = get_possible_actions(st)
            if st not in returns.keys():
                returns[st] = np.zeros(action_dim)
                returns_count[st] = np.zeros(action_dim)
                q[st] = np.zeros(action_dim)
                pi[st]= np.zeros(action_dim)

                for a in range(action_dim):
                    if a not in possible_actions:
                        q[st][a] = -999999

            returns[st][at] += G
            returns_count[st][at] += 1
            q[st][at] = returns[st][at] / returns_count[st][at]
            pi[st]= np.zeros(action_dim)
            
            pi[st][np.argmax(q[st])] = 1.0
        if eval_results:
        	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
        	results_file.write(f"{episode_id} {successes} {fails} {steps} {rewards}\n")
    if eval_results:
    	results_file.close()
    return q, pi



def on_policy_first_visit_monte_carlo_control(
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 1000,
        max_steps_per_episode: int = 100,
        epsilon: float = 0.2,
        eval_results: bool = True,
        gamma: float = 0.99,
        action_dim: int = 9,
) -> (np.ndarray, np.ndarray):

    pi = {}

    q = {}
    if eval_results:
    	pi_ = {}
    	directory = 'Data/'
    	if not os.path.exists(directory):
    		os.makedirs(directory)
    	results_file = open(directory+'on_policy_first_visit_monte_carlo_control_results.txt', "a") 

    returns = {} # np.zeros((states_count, actions_count))
    returns_count = {} # np.zeros((states_count, actions_count))

    for episode_id in range(episodes_count):
        s0 = reset_func()

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_return_history(s0, pi, is_terminal_func,
                                                                                         step_func, get_possible_actions,
                                                                                         max_steps_per_episode,
                                                                                         action_dim = action_dim)

        G = 0
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]

            if (st, at) in zip(s_list[0:t], a_list[0:t]):
                continue
            if st not in returns.keys():
                returns[st] = np.zeros(action_dim)
                returns_count[st] = np.zeros(action_dim)
                q[st] = np.zeros(action_dim)

            returns[st][at] += G
            returns_count[st][at] += 1
            # print(q[st])
            q[st][at] = returns[st][at] / returns_count[st][at]
            possible_actions = get_possible_actions(st)
            pi[st] = np.ones(action_dim) * (epsilon / len(possible_actions))
            for a in range(action_dim):
                if a not in possible_actions:
                    pi[st][a] = 0
                    q[st][a] = -999999
            pi[st][np.argmax(q[st])] = 1.0 - epsilon + epsilon / len(possible_actions)
            if eval_results:
            	pi_[st] = np.zeros(action_dim)
            	pi_[st][np.argmax(q[st])] = 1.0 
            	successes, fails, steps, rewards = eval(pi_, reset_func, step_func, get_possible_actions, action_dim = action_dim)
            	results_file.write(f"{episode_id} {successes} {fails} {steps} {rewards}\n")

    if eval_results:
    	results_file.close()

    return q, pi




def off_policy_monte_carlo_control(
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 1000,
        max_steps_per_episode: int = 100,
        epsilon: float = 0.2,
        eval_results: bool = True,
        gamma: float = 0.99,
        action_dim: int = 9,
) -> (np.ndarray, np.ndarray):


    b = {}
    pi = {}
    C = {}
    q = {}
    if eval_results:
    	directory = 'Data/'
    	if not os.path.exists(directory):
    		os.makedirs(directory)
    	results_file = open(directory+'off_policy_monte_carlo_control_results.txt', "a") 

    for episode_id in range(episodes_count):
        s0 = reset_func()

        s_list, a_list, _, r_list = step_until_the_end_of_the_episode_and_return_history(s0, b, is_terminal_func,
                                                                                         step_func, get_possible_actions,
                                                                                         max_steps_per_episode,
                                                                                         action_dim = action_dim)

        G = 0
        W = 1
        for t in reversed(range(len(s_list))):
            G = gamma * G + r_list[t]
            st = s_list[t]
            at = a_list[t]
            possible_actions = get_possible_actions(st)
            if st not in C.keys():
                C[st] = np.zeros(action_dim)
                q[st] = np.zeros(action_dim)
                pi[st] = np.zeros(action_dim)
                b[st]=np.ones(action_dim)*1.0/len(possible_actions)
                for a in range(action_dim):
                    if a not in possible_actions:
                        q[st][a] = -9999999
                        b[st][a] = 0
            C[st][at] += W

            q[st][at] += W / C[st][at] * (G - q[st][at])
            pi[st] = np.zeros(action_dim)
            pi[st][np.argmax(q[st])] = 1.0

            if at != np.argmax(q[st]):
                break

            W = W / b[st][at]
        if eval_results:
        	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
        	results_file.write(f"{episode_id} {successes} {fails} {steps} {rewards}\n")

    if eval_results:
    	results_file.close()

    return q, pi

def tabular_td_zero_prediction(
        pi: np.ndarray,
        is_terminal_func: Callable,
        reset_func: Callable,
        step_func: Callable,
        episodes_count: int = 1000,
        eval_results: bool = True,
        max_steps_per_episode: int = 100,
        gamma: float = 0.99,
        alpha: float = 0.01,
) -> np.ndarray:
    states = np.arange(pi.shape[0])
    actions = np.arange(pi.shape[1])
    V = np.random.random(pi.shape[0])
    if eval_results:
    	directory = 'Data/'
    	if not os.path.exists(directory):
    		os.makedirs(directory)
    	results_file = open(directory+'tabular_TD0_results.txt', "a") 
    for s in states:
        if is_terminal_func(s):
            V[s] = 0

    for episode_id in range(episodes_count):
        s = reset_func()

        step = 0
        while not is_terminal_func(s) and step < max_steps_per_episode:
            a = np.random.choice(actions, p=pi[s])
            (s_p, r, t) = step_func(s, a)
            V[s] += alpha * (r + gamma * V[s_p] - V[s])
            s = s_p
            step += 1

    return V



def tabular_sarsa_control(
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 1000,
        max_steps_per_episode: int = 100,
        eval_results: bool = True,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        gamma: float = 0.99,
        action_dim: int = 9,
) -> (np.ndarray, np.ndarray):

    q = {} # np.random.random((states_count, actions_count))
    if eval_results:
    	directory = 'Data/'
    	if not os.path.exists(directory):
    		os.makedirs(directory)
    	results_file = open(directory+'tabular_sarsa_results.txt', "a") 

    for episode_id in range(episodes_count):

        s = reset_func()
        rdm = np.random.random()
        if s not in q.keys():
            q[s]=np.random.random(action_dim)

        possible_actions = get_possible_actions(s)

        for action in range(action_dim):
            if action not in possible_actions:
                q[s][action] = -99999

        a = np.random.choice(possible_actions) if rdm < epsilon else np.argmax(q[s])
        step = 0
        # print(s, possible_actions, q[s])
        while not is_terminal_func(s) and step < max_steps_per_episode:
            s_p, r, done,_ = step_func(a)

            rdm = np.random.random()
            if s_p not in q.keys():
                q[s_p] = np.random.random(action_dim)
                possible_actions = get_possible_actions(s_p)
            for action in range(action_dim):
                if action not in possible_actions:
                    q[s_p][action] = -99999
                if is_terminal_func(s_p):
                    q[s_p][action] = 0


            possible_actions = get_possible_actions(s_p)
            if len(possible_actions) == 0:
                possible_actions = range(action_dim)
            a_p = np.random.choice(possible_actions) if rdm < epsilon else np.argmax(q[s_p])
            q[s][a] += alpha * (r + gamma * q[s_p][a_p] - q[s][a])
            
            step += 1
            s = s_p
            a = a_p

        if eval_results:
            	pi = {} # np.zeros_like(q)
            	for s in q.keys():
            		possible_actions = get_possible_actions(s)
            		if not is_terminal_func(s):
            			pi[s] = np.zeros(action_dim)
            			pi[s][np.argmax(q[s])] = 1.0 
            	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
            	results_file.write(f"{episode_id} {successes} {fails} {steps} {rewards}\n")


    if eval_results:
    	results_file.close()
    else:
	    pi = {} # np.zeros_like(q)
	    for s in q.keys():
	        possible_actions = get_possible_actions(s)
	        if not is_terminal_func(s):
	            pi[s] = np.ones(action_dim) * (epsilon / len(possible_actions))
	            for action in range(action_dim):
	                if action not in possible_actions:
	                    pi[s][action] = 0
	            pi[s][np.argmax(q[s])] = 1.0 - epsilon + epsilon / len(possible_actions)

    return q, pi

def tabular_expected_sarsa_control(
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 1000,
        max_steps_per_episode: int = 100,
        epsilon: float = 0.1,
        eval_results: bool = True,
        alpha: float = 0.1,
        gamma: float = 0.99,
        action_dim: int = 9,
) -> (np.ndarray, np.ndarray):
    q = {}
    if eval_results:
    	directory = 'Data/'
    	pi = {}
    	if not os.path.exists(directory):
    		os.makedirs(directory)
    	results_file = open(directory+'expected_sarsa_results.txt', "a") 
    	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
    	results_file.write(f"0 {successes} {fails} {steps} {rewards}\n")

    for episode_id in range(episodes_count):

        s = reset_func()
        if s not in q.keys():
            q[s]=np.random.random(action_dim)
        step = 0

        while not is_terminal_func(s) and step < max_steps_per_episode:
            rdm = np.random.random()
            possible_actions = get_possible_actions(s)

            for action in range(action_dim):
                if action not in possible_actions:
                    q[s][action] = -99999
                if is_terminal_func(s):
                    q[s][action] = 0

            a = np.random.choice(possible_actions) if rdm < epsilon else np.argmax(q[s])
            s_p, r, done,_ = step_func(a)
            
            if s_p not in q.keys():
                q[s_p] = np.random.random(action_dim)
                possible_actions = get_possible_actions(s_p)

            for action in range(action_dim):
                if action not in get_possible_actions(s_p):
                    q[s_p][action] = 0
                if is_terminal_func(s_p):
                    q[s_p][action] = 0

            possible_actions = get_possible_actions(s_p)
            if len(possible_actions) == 0:
                possible_actions = range(action_dim)
            expected_value = np.sum(q[s_p])*1.0/len(possible_actions)

            q[s][a] += alpha * (r + gamma * expected_value - q[s][a])
            s = s_p
            step += 1

        if eval_results:
            	pi = {} # np.zeros_like(q)
            	for s in q.keys():
            		possible_actions = get_possible_actions(s)
            		if not is_terminal_func(s):
            			pi[s] = np.zeros(action_dim)
            			for action in range(action_dim):
            				if action not in possible_actions:
            					pi[s][action] = 0
            					q[s][action] = -999999
            			pi[s][np.argmax(q[s])] = 1.0 
            	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
            	results_file.write(f"{episode_id} {successes} {fails} {steps} {rewards}\n")

    if not eval_results:
	    pi = {} 
	    for s in q.keys():
	        possible_actions = get_possible_actions(s)
	        if not is_terminal_func(s):
	            pi[s] = np.ones(action_dim) * (epsilon / len(possible_actions))
	            for action in range(action_dim):
	                if action not in possible_actions:
	                    pi[s][action] = 0
	                    q[s][action] = -999999
	            pi[s][np.argmax(q[s])] = 1.0 - epsilon + epsilon / len(possible_actions)
    else:
    	results_file.close()

    return q, pi


def tabular_q_learning_control(
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 1000,
        max_steps_per_episode: int = 100,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        gamma: float = 0.99,
        eval_results: bool = True,
        action_dim = 9
) -> (np.ndarray, np.ndarray):

    q = {} 
    pi = {}
    if eval_results:
    	directory = 'Data/'
    	if not os.path.exists(directory):
    		os.makedirs(directory)
    	results_file = open(directory+'tabular_q_learning_results.txt', "a")
    	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
    	results_file.write(f"0 {successes} {fails} {steps} {rewards}\n")

    for episode_id in range(episodes_count):
        s = reset_func()
        step = 0
        

        while not is_terminal_func(s) and step < max_steps_per_episode:

            if s not in q.keys():
                q[s]=np.random.random(action_dim)
            possible_actions = get_possible_actions(s)

            for action in range(action_dim):
                if action not in possible_actions:
                    q[s][action] = -999999

            rdm = np.random.random()
            possible_actions = get_possible_actions(s)
            a = np.random.choice(possible_actions) if rdm < epsilon else np.argmax(q[s])
            (s_p, r, t, _) = step_func(a)
            if s_p not in q.keys():
                q[s_p]=np.random.random(action_dim)
            possible_actions = get_possible_actions(s_p)

            for action in range(action_dim):  
                if action not in possible_actions:
                    q[s_p][action] = -999999
            q[s][a] += alpha * (r + gamma * np.max(q[s_p]) - q[s][a])
            s = s_p
            step += 1
        if eval_results:
            pi = {}
            for s in q.keys():
                pi[s] = np.zeros(action_dim)
                pi[s][np.argmax(q[s])] = 1.0
            successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
            results_file.write(f"{episode_id} {successes} {fails} {steps} {rewards}\n")

    if not eval_results:
        pi = {}
        for s in q.keys():
            pi[s] = np.zeros(action_dim)
            pi[s][np.argmax(q[s])] = 1.0
    else:
    	results_file.close()
    

    return q, pi, None




def dyna_q_control(
        reset_func: Callable,
        is_terminal_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 10000,
        max_steps_per_episode: int = 100,
        eval_results: bool = True,
        epsilon: float = 0.2, # to explore , 
        alpha: float = 0.1,
        gamma: float = 0,
        n: int = 10,
        action_dim = 9
) -> (np.ndarray, np.ndarray):
    q = {} 
    model = {}
    pi = {}
    if eval_results:
    	directory = 'Datahp/'
    	if not os.path.exists(directory):
    		os.makedirs(directory)
    	results_file = open(directory+'dyna_q_control_results.txt', "a")
    	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
    	results_file.write(f"0 {successes} {fails} {steps} {rewards}\n")  

    for episode_id in range(episodes_count):
        s = reset_func()
        if s not in q.keys():
            q[s]=np.random.random(action_dim)

        possible_actions = get_possible_actions(s)

        for action in range(action_dim):
            if action not in possible_actions:
                q[s][action] = -99999

        
        step = 0
        while not is_terminal_func(s) and step < max_steps_per_episode:

            rdm = np.random.random()
            possible_actions = get_possible_actions(s)
            a = np.random.choice(possible_actions) if rdm < epsilon else np.argmax(q[s])
            s_p, r, done, _ = step_func(a)
            
            model[(s, a)] = (s_p, r)



            if s_p not in q.keys():
                q[s_p] = np.random.random(action_dim)
                possible_actions = get_possible_actions(s_p)

                for action in range(action_dim):
                    if action not in possible_actions:
                        q[s_p][action] = -99999

                    if is_terminal_func(s_p):
                        q[s_p][action] = 0


            delta = r + gamma * q[s_p].max() - q[s][a]
            q[s][a] += alpha * delta
            s = s_p
            step += 1

            # perform n steps
            for _ in range(n):
                (s_,a_), (s_p, r) = random.choice(list(model.items()))

                # update
                delta = r + gamma * q[s_p].max() - q[s_][a_]
                q[s_][a_] += alpha * delta
        if eval_results:
        	pi = {} # np.zeros_like(q)
        	for s in q.keys():
        		possible_actions = get_possible_actions(s)
        		if not is_terminal_func(s):
        			pi[s] = np.zeros(action_dim)
        			pi[s][np.argmax(q[s])] = 1.0
        	successes, fails, steps, rewards = eval(pi, reset_func, step_func, get_possible_actions, action_dim = action_dim)
        	results_file.write(f"{episode_id} {successes} {fails} {steps} {rewards}\n") 



    if eval_results:
    	results_file.close()
    else:
	    pi = {} # np.zeros_like(q)
	    for s in q.keys():
	        possible_actions = get_possible_actions(s)
	        if not is_terminal_func(s):
	            pi[s] = np.zeros(action_dim) 

	            pi[s][np.argmax(q[s])] = 1.0 

    return q, pi


def reinforce(reset_func: Callable,
        step_func: Callable,
        get_possible_actions: Callable,
        episodes_count: int = 10000,
        max_steps_per_episode: int = 100,
        eval_results: bool = True,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        gamma: float = 0.99,
        state_dim = 1,
        action_dim = 4):

    weight = np.random.rand(state_dim, action_dim)
    episode_rewards = []

    if eval_results:
    	directory = 'Data/'
    	if not os.path.exists(directory):
    		os.makedirs(directory)
    	results_file = open(directory+'reinforce_results.txt', "w") 

    for episode_id in range(episodes_count):

    	s = reset_func()[None, :]
    	rewards = []
    	gradients = []
    	t = 0
    	done = False

    	while not done and t < max_steps_per_episode:
    		t += 1
    		#print(s)
    		Pi  = softmax_policy(weight, s)
    		a = np.random.choice(action_dim,p=Pi[0])
    		s_p, r, done, _ = step_func(a)
    		

    		rewards.append(r)
    		s_p = s_p
    		s_p = s_p[None, :]
    		dl = softmax_gradient(Pi)[a, :] / Pi[0, a]
    		gradients.append(s.T.dot(dl[None, :]))
    		s = s_p
    	if eval_results:
    		results_file.write(f'{t} \n')

    	# Update the weight
    	for i in range(len(rewards)):
    		tab = []
    		for t , r in enumerate(rewards[i:]):
    			tab += [r * gamma ** t]
    		weight += alpha * gradients[i] * sum(tab)
    if eval_results:
    	results_file.close()
    return Pi


