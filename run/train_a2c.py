"""
@author: Saurabh Kumar
"""

import os
import sys

sys.path.insert(0, '../rl_agents/policy_gradient/')
sys.path.insert(0, '../rl_agents/utils/')
sys.path.insert(0, 'environments/')
sys.path.insert(0, 'experiment_logs/')


import a2c
import numpy as np
import objects_env
import reinforce


def log(logfile, iteration, rewards):
	"""Function that logs the reward statistics obtained by the agent.

	Args:
		logfile: File to log reward statistics. 
		iteration: The current iteration.
		rewards: Array of rewards obtained in the current iteration.
	"""
	log_string = '{} {} {} {}'.format(
		iteration, np.min(rewards), np.mean(rewards), np.max(rewards))
	print(log_string)

	with open(logfile, 'a') as f:
		f.write(log_string + '\n')


def make_environment(env_name):
	if env_name == 'objects_environment':
		return objects_env.World()
	else:
		return None


def make_agent(agent_type, env):
	if agent_type == 'reinforce':
		return reinforce.ReinforceAgent(state_dims=env.get_state_size(),
					   					num_actions=env.get_num_actions(),
					   					network_type='conv', 
					   					baseline_type='critic',
					   					entropy_regularization=True)
	elif agent_type == 'a2c':
		return a2c.A2CAgent(state_dims=env.get_state_size(),
					   		num_actions=env.get_num_actions(),
					   		network_type='conv',
					   		baseline_type='critic',
					   		entropy_regularization=True)
	else:
		return None


def run(env_name='objects_environment', 
		agent_type='a2c', 
		num_iterations=100, 
		num_train_episodes=10, 
		num_eval_episodes=10,
		logfile=None):
	"""Function that executes RL training and evaluation.

	Args:
		env_name: Name of the environment that the agent will interact with. 
		agent_type: The type RL agent that will be used for training.
		num_iterations: Number of iterations to train for.
		num_train_episodes: Number of training episodes per iteration.
		num_eval_episodes: Number of evaluation episodes per iteration.
		logfile: File to log the agent's performance over training.
	"""

	env = make_environment(env_name)
	agent = make_agent(agent_type, env)

	for it in range(num_iterations):

		# if it % 1 == 0:
		#	agent.save_checkpoint(it)

		# Run train episodes.

		for train_episode in range(num_train_episodes):

			# Reset the environment.
			state = env.reset()
			state = np.expand_dims(state, axis=0)

			agent.reset()
			episode_reward = 0

			# Run the episode.
			terminal = False
			while not terminal:

				action = agent.sample(state)
				action_probs = agent.get_action_probs(state)

				print(action_probs)
				print action

				next_state, reward, terminal = env.step(action)
				next_state = np.expand_dims(next_state, axis=0)

				episode_reward += reward

				# Update the agent based on the current transition.
				agent.update(state, action, reward, next_state, terminal)

				# Update the state.
				state = next_state

		eval_rewards = []

		for eval_episode in range(num_eval_episodes):

			# Reset the environment.
			state = env.reset()
			state = np.expand_dims(state, axis=0)

			episode_reward = 0

			# Run the episode. 
			terminal = False

			while not terminal:
				action = agent.best_action(state)

				next_state, reward, terminal = env.step(action)
				next_state = np.expand_dims(next_state, axis=0)

				episode_reward += reward

				state = next_state

			eval_rewards.append(episode_reward)

		log(logfile, it, eval_rewards)


logfile = 'experiment_logs/exp1.txt'
env_name = 'objects_environment'
agent_type ='a2c'
run(env_name=env_name, agent_type=agent_type, logfile=logfile)