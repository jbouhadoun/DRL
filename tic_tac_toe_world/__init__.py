import numpy as np
import random

nb_cells = 9 # Grille 3x3 

marks = {0:'X', 1:'O'}

"""
L'action i correspond à cocher la case i. Les cases sont numérotées comme suit:

| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 | 8 | 9 |

"""

current_state = {}
player_id = 0
human_player = False

def get_possible_actions(state):
	possible_actions = []
	for k in range(len(state)):
		if state[k]==' ':
			possible_actions.append(k)

	return possible_actions


def reset (human = False) -> dict:
	global player_id
	global current_state
	global human_player

	player_id = random.randint(0,1) # Choisir le joueur qui commence la partie (joueur 0 ou 1)
	human_player = human
	for i in range(nb_cells):
		current_state[i] = ' '

	if player_id == 1:
		if not human_player:
			action = random.choice(get_possible_actions(current_state))
		else:
			render()
			action = int(input("Choose action"))

		current_state[action]=marks[player_id]
		if human_player:
			render()
		player_id = 0


	return array_to_str(current_state)

def render():
	global current_state

	for i in range(3):
		print("| ", end = '')
		for j in range(3):
			if j == 2: 
				end = '\n'
			else:
				end = ""
			print(current_state[i*3+j]+" | ", end=end)
	print()

def array_to_str(state):
	str_ = ''
	for i in range(len(state)):
		str_+=state[i]
	return str_




def is_successfull(state):
	for i in range(3):
		if state[i]==state[i+3]==state[i+6] and state[i] in marks.values():
			return True

		if state[3*i]==state[3*i+1]==state[3*i+2] and state[3*i] in marks.values():
			return True

	if state[0]==state[4]==state[8] and state[0] in marks.values():
		return True

	if state[2]==state[4]==state[6] and state[2] in marks.values():
		return True

	return False

def is_terminal(state):
	return is_successfull(state) or len(get_possible_actions(state)) == 0




def step(action):
	global player_id
	global current_state
	global human_player

	assert current_state[action] == ' ' and not is_terminal(current_state)
	current_state[action] = marks[player_id] # Le joueur 'player_id' coche la case 'action'
	
	done = is_terminal(current_state)
	#assert player_id==0

	if done:
		if not is_successfull(current_state):
			reward = 0

		else:
			reward = 1


	else:
		player_id = ( player_id + 1 ) % 2 # On passe la main à l'autre joueur
		reward = 0

		if not human_player:
			action = random.choice(get_possible_actions(current_state))
		else:
			render()
			action = int(input("Choose action"))
			

		current_state[action] = marks[player_id]
		player_id = 0

		done = is_terminal(current_state)
			
		if done:
			if not is_successfull(current_state):
				reward = 0
			else:
				reward = -1

	if human_player:
		render()

	return array_to_str(current_state), reward, done

