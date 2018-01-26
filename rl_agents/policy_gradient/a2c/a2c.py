"""
@author: Saurabh Kumar
"""

import numpy as np
import os
import reinforce
import sys
import tensorflow.contrib.layers as layers
import tensorflow as tf

from utils import weight_variable, bias_variable, conv2d


class A2CAgent(reinforce.ReinforceAgent):

	def __init__(self, *args, **kwargs):
		super(A2CAgent, self).__init__(*args, **kwargs)

	def update(self, state, action, reward, next_state, terminal):
		td_target = reward + (1 - terminal) * self.DISCOUNT * self._get_baseline(next_state)
		td_error = td_target - self._get_baseline(state)

		'''
		print('TD Target:')
		print(td_target)
		print('Current state value:')
		print(self._get_baseline(state))
		print('TD Error:')
		print(td_error)
		print('')
		'''
		self._update_actor(state, action, td_error)
		self._update_critic(state, td_target)