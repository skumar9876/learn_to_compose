"""
@author: Saurabh Kumar
"""

import numpy as np
import os
import sys
import tensorflow.contrib.layers as layers
import tensorflow as tf

from utils import weight_variable, bias_variable, conv2d, Transition


class ReinforceAgent(object):

	DISCOUNT = 0.95
	NUM_CONV_1_FILTERS = 10
	NUM_CONV_2_FILTERS = 20
	BASELINE_TYPES = ['None', 'critic']
	NETWORK_TYPES = ['conv', 'feedforward']
	ENTROPY_REG = 0.05

	def __init__(self, 
				 initial_policy_lr=0.001, 
				 initial_critic_lr=0.01,
				 state_dims=[],
				 num_actions=0,
				 summary_dir=None, 
				 sess=None, 
				 baseline_type=None, 
				 network_type=None,
				 entropy_regularization=False):
		self._policy_initial_lr = initial_policy_lr
		self._critic_initial_lr = initial_critic_lr
		self._state_dims = state_dims
		self._num_actions = num_actions
		self._summary_dir = summary_dir
		# Which kind of baseline to use.
		self._baseline_type = baseline_type
		# Which type of network to use: convolutional or feedforward.
		self._network_type = network_type
		# Whether or not to use entropy regularization in policy loss.
		self._entropy_regularization = entropy_regularization

		self._episode_transitions = []

		assert self._baseline_type in self.BASELINE_TYPES
		assert self._network_type in self.NETWORK_TYPES

		with tf.Graph().as_default():
			self._construct_graph()
			self._saver = tf.train.Saver()
			if sess is None:
				self.sess = tf.Session()
			else:
				self.sess = sess
			self.sess.run(tf.global_variables_initializer())

	def _policy_network(self, state):
		"""Builds the policy network."""

		if self._network_type == 'conv':
			policy_w1 = weight_variable([3, 3, 1, self.NUM_CONV_1_FILTERS], name='w1')
			policy_b1 = bias_variable([self.NUM_CONV_1_FILTERS])
			policy_conv1 = tf.nn.relu(conv2d(state, policy_w1) + policy_b1)

			policy_w2 = weight_variable(
				[3, 3, self.NUM_CONV_1_FILTERS, self.NUM_CONV_2_FILTERS], name='w2')
			policy_b2 = bias_variable([self.NUM_CONV_2_FILTERS])
			policy_conv2 = tf.nn.relu(conv2d(policy_conv1, policy_w2) + policy_b2)

			policy_flattened = layers.flatten(policy_conv2)

			policy_w3 = weight_variable(
				[policy_flattened.get_shape().as_list()[1], self._num_actions], name='w3')
			policy_b3 = bias_variable([self._num_actions])
			action_out = tf.matmul(policy_flattened, policy_w3) + policy_b3

		elif self._network_type == 'feedforward':
			state_flattened = layers.flatten(state)
			policy_w1 = weight_variable(
				[state_flattened.get_shape().as_list()[1], 100], name='w1')
			policy_b1 = bias_variable([100])
			policy_ff1 = tf.tanh(tf.matmul(state_flattened, policy_w1) + policy_b1)

			policy_w2 = weight_variable([100, 100], name='w2')
			policy_b2 = bias_variable([100])
			policy_flattened = tf.tanh(tf.matmul(policy_ff1, policy_w2) + policy_b2)

			policy_w3 = weight_variable(
				[policy_flattened.get_shape().as_list()[1], self._num_actions], name='w3')
			policy_b3 = bias_variable([self._num_actions])
			action_out = tf.matmul(policy_flattened, policy_w3) + policy_b3

		action_probs = tf.nn.softmax(action_out)

		return action_out, action_probs

	def _value_network(self, state):
		"""Builds the value network."""
		w1 = weight_variable([3, 3, 1, self.NUM_CONV_1_FILTERS], name='w1')
		b1 = bias_variable([self.NUM_CONV_1_FILTERS])
		conv1 = tf.nn.relu(conv2d(state, w1) + b1)

		w2 = weight_variable(
			[3, 3, self.NUM_CONV_1_FILTERS, self.NUM_CONV_2_FILTERS], name='w2')
		b2 = bias_variable([self.NUM_CONV_2_FILTERS])
		conv2 = tf.nn.relu(conv2d(conv1, w2) + b2)

		value_flattened = layers.flatten(conv2)

		w3 = weight_variable([value_flattened.get_shape().as_list()[1], 1], name='w3')
		b3 = bias_variable([1])
		value = tf.matmul(value_flattened, w3) + b3

		return value

	def _construct_graph(self):
		"""Builds the tensorflow graph ops for the policy and critic networks along with their updates."""
		self._state = tf.placeholder(
			shape=[None, self._state_dims[0], self._state_dims[1], 1], dtype=tf.float32)

		self._state_normalized = tf.to_float(self._state) / 255.0

		with tf.variable_scope('policy_network'):
			self._action_out, self._action_probs = self._policy_network(self._state_normalized)
			self._action_entropy = -tf.reduce_mean(
				tf.reduce_sum(self._action_probs * tf.log(self._action_probs), 1), 0)

		with tf.variable_scope('policy_update'):
			self._picked_action = tf.placeholder(dtype=tf.int32)
			self._policy_target = tf.placeholder(dtype=tf.float32)
			self._picked_action_prob = tf.gather_nd(self._action_probs, 
				[[0, self._picked_action]])

			self._policy_loss_no_reg = tf.reduce_mean(
				-tf.log(self._picked_action_prob) * self._policy_target) 

			if self._entropy_regularization:
				self._policy_loss = self._policy_loss_no_reg - self.ENTROPY_REG * self._action_entropy
			else:
				self._policy_loss = self._policy_loss_no_reg

			self._policy_optimizer = tf.train.AdamOptimizer(
				learning_rate=self._policy_initial_lr)

			policy_grads_and_vars = self._policy_optimizer.compute_gradients(self._policy_loss)
			policy_clipped_grads_and_vars = [(
				tf.clip_by_norm(grad, 5.0), var) for grad, var in policy_grads_and_vars if grad is not None]

			self._policy_train_op = self._policy_optimizer.apply_gradients(
				policy_clipped_grads_and_vars, global_step=tf.contrib.framework.get_global_step())

		with tf.variable_scope('critic_network'):
			self._value = self._value_network(self._state_normalized)

		with tf.variable_scope('critic_update'):
			self._value_target = tf.placeholder(tf.float32)
			self._critic_loss = tf.reduce_mean(tf.squared_difference(
				self._value, self._value_target))

			self._critic_optimizer = tf.train.AdamOptimizer(
				learning_rate=self._critic_initial_lr)

			critic_grads_and_vars = self._critic_optimizer.compute_gradients(self._critic_loss)
			critic_clipped_grads_and_vars = [(
				tf.clip_by_norm(grad, 5.0), var) for grad, var in critic_grads_and_vars if grad is not None]

			self._critic_train_op = self._critic_optimizer.apply_gradients(
				critic_clipped_grads_and_vars, global_step=tf.contrib.framework.get_global_step())

	def _update_actor(self, state, action, td_error):
		feed_dict = { self._state: state, 
					  self._picked_action: action, 
					  self._policy_target: td_error }

		_ = self.sess.run(self._policy_train_op, feed_dict)

	def _update_critic(self, state, td_target):
		feed_dict = { self._state: state, 
					  self._value_target: td_target }

		_ = self.sess.run(self._critic_train_op, feed_dict)

	def _get_baseline(self, state):
		if self._baseline_type == 'None':
			return 0
		if self._baseline_type == 'critic':
			feed_dict = { self._state: state }
			return self.sess.run(self._value, feed_dict=feed_dict)[0][0]

	def _update_baseline(self, state, td_target):
		if self._baseline_type == 'critic':
			self._update_critic(state, td_target)

	def get_action_out_weights(self):
		return self._policy_vars[-2]

	def get_action_out_biases(self):
		return self._policy_vars[-1]

	def update(self, state, action, reward, next_state, terminal):
		
		self._episode_transitions.append(
			Transition(state, action, reward, next_state, terminal))

		# Loop through the episode.
		# Compute discounted return from each state until the episode termination.
		# Use this computation to update both the actor and baseline.
		if terminal:
			discounted_return = 0
			for transition in reversed(self._episode_transitions):
				discounted_return = self.DISCOUNT * discounted_return + transition.reward

				baseline = self._get_baseline(transition.state)
				td_error = discounted_return - baseline

				self._update_actor(transition.state, transition.action, td_error)
				self._update_baseline(transition.state, discounted_return)

	def get_action_probs(self, state):
		"""Returns the probability distribution over actions for the given state as input."""
		feed_dict = { self._state: state }
		action_probs = self.sess.run(self._action_probs, feed_dict=feed_dict)[0]
		return action_probs

	def sample(self, state):
		"""Returns action sampled according to the agent's policy."""
		action_probs = self.get_action_probs(state)
		return np.random.choice(np.arange(len(action_probs)), p=action_probs)

	def best_action(self, state):
		"""Returns the action with the highest probability in the agent's policy."""
		feed_dict = { self._state: state }
		action_probs = self.sess.run(self._action_probs, feed_dict=feed_dict)
		return np.argmax(action_probs)

	def reset(self):
		print("Reset!")
		self._episode_transitions = []

	def save_checkpoint(self, iteration):
		tf.add_to_collection('policy_train_op', self._policy_train_op)
		tf.add_to_collection('action_probs', self._action_probs)
		tf.add_to_collection('action_entropy', self._action_entropy)
		tf.add_to_collection('state', self._state)

		self._saver.save(self.sess, os.path.join(
			os.getcwd(), 'experiment_logs/checkpoints/model.ckpt'), global_step=iteration)
