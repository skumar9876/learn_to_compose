"""
@author: Saurabh Kumar
"""

import a2c
import numpy as np
import os
import sys
import tensorflow.contrib.layers as layers
import tensorflow as tf

from utils import weight_variable, bias_variable, conv2d


class SharedA2CAgent(object):

	def __init__(self, 
				 initial_policy_lr=0.001, 
				 initial_critic_lr=0.01,
				 state_dims=[],
				 num_actions=0,
				 summary_dir=None):

		self.agents = [None, None]
		self.action_out_weights = [None, None]
		self.action_out_biases = [None, None]
		self.sess = tf.Session()

		for i in xrange(len(agents)):
			self.agents[i] = a2c.A2CAgent(initial_policy_lr, initial_critic_lr, 
									      state_dims, num_actions, summary_dir, 
									      self.sess)
			self.action_out_weights = self.agents[i].get_action_out_weights()
			self.action_out_biases = self.agents[i].get_action_out_biases()

		with tf.name_scope('copy_action_out_weights_and_biases'):
			self.task_id = tf.placeholder(shape=(), dtype=tf.int32)
			from_index = self.task_id
			to_index = (self.task_id + 1) % 2

			self.copy_weights = self._action_out_weights[to_index].assign(
				self.action_out_weights[from_index])
			self.copy_biases = self._action_out_biases[to_index].assign(
				self.action_out_biases[from_index])

	def update(self, state, action, reward, next_state, terminal, task_id=0):
		self.agents[task_id].update(state, action, reward, next_state, terminal)

		feed_dict = { self.task_id: task_id }
		self.sess.run([self.copy_weights, self.copy_biases], feed_dict=feed_dict)

	def get_action_probs(self, state, task_id):
		return self.agents[task_id].get_action_probs(state, task_id)

	def sample(self, state, task_id):
		return self.agents[task_id].sample(state)

	def best_action(self, state, task_id):
		return self.agents[task_id].best_action(state)

	def save_checkpoint(self, iteration):
		# @TODO(saurabh): Figure out how to save checkpoints properly!
		# tf.add_to_collection('policy_train_op', self._policy_train_op)
		# tf.add_to_collection('action_probs', self._action_probs)
		# tf.add_to_collection('action_entropy', self._action_entropy)
		# tf.add_to_collection('state', self._state)

		# self._saver.save(self.sess, os.path.join(os.getcwd(), 'experiment_logs/checkpoints/model.ckpt'), global_step=iteration)
		pass
