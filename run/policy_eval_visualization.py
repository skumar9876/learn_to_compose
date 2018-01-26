"""
@author: dennybritz

Modified for the ComposeNet project by Saurabh Kumar.

Temporary file that does policy evaluation + visualization.
"""

import sys
import os
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import tensorflow as tf
import time

from estimators import ValueEstimator, PolicyEstimator
from worker import make_copy_params_op


def make_plot(curr_map, agent, objects, obj1, obj2, obj3, writer):
  objects_values = [255.0/3, 2*255.0/3, 3*255.0/3]

  agent_x = []
  agent_y = []

  for obj in objects:
    obj.set_data([], [])

  for i in xrange(len(curr_map)):
    for j in xrange(len(curr_map[i])):

      map_val = curr_map[i][j]

      if map_val == objects_values[0]:
        obj1.set_data([j], [i])

      elif map_val == objects_values[1]:
        obj2.set_data([j], [i])

      elif map_val == objects_values[2]:
        obj3.set_data([j], [i])

      elif map_val == 10:
        agent.set_data([j], [i])

  writer.grab_frame()


class VisualizePolicy(object):
  """
  Helps evaluating a policy by running a fixed number of episodes in an environment,
  and recording a video of the policy executing over those episodes.
  Args:
    env: environment to run in
    policy_net: A policy estimator
  """
  def __init__(self, env, policy_net, task):

    self.env = env
    self.global_policy_net = policy_net
    self.task = task

    # Local policy net
    with tf.variable_scope("policy_visualization"):
      self.policy_net = PolicyEstimator(policy_net.num_outputs, state_dims=self.env.get_state_size())

    # Op to copy params from global policy/value net parameters
    self.copy_params_op = make_copy_params_op(
      tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
      tf.contrib.slim.get_variables(scope="policy_visualization", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

  def _policy_net_predict(self, state, sess):
    feed_dict = { self.policy_net.states: [state] }
    preds = sess.run(self.policy_net.predictions, feed_dict)
    return preds["probs"][0]

  def eval(self, sess):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Agent Evaluation', artist='Matplotlib',
                  comment='Evaluation!')
    writer = FFMpegWriter(fps=3, metadata=metadata)

    fig = plt.figure()
    agent, = plt.plot([], [], 'ko', markersize=20)
    obj1, = plt.plot([], [], 'b*', markersize=20)
    obj2, = plt.plot([], [], 'r*', markersize=20)
    obj3, = plt.plot([], [], 'y*', markersize=20)

    objects = [obj1, obj2, obj3]

    plt.xlim(-1, 5)
    plt.ylim(-1, 5)

    with sess.as_default(), sess.graph.as_default():
      # Copy params to local model
      global_step, _ = sess.run([tf.contrib.framework.get_global_step(), self.copy_params_op])

      eval_rewards = []
      episode_lengths = []

      with writer.saving(fig, '{}.mp4'.format(self.task), 100):
        for i in xrange(10):
          # Run an episode
          done = False
          curr_map, state = self.env.reset_test()
          make_plot(curr_map, agent, objects, obj1, obj2, obj3, writer)

          total_reward = 0.0
          episode_length = 0

          while not done:
            action_probs = self._policy_net_predict(state, sess)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_map, next_state, reward, done = self.env.step_test(action)
            make_plot(next_map, agent, objects, obj1, obj2, obj3, writer)

            total_reward += reward
            episode_length += 1

            state = next_state
            curr_map = next_map

          eval_rewards.append(total_reward)
          episode_lengths.append(episode_length)

  def continuous_eval(self, eval_every, sess, coord):
    """
    Continuously evaluates and visualizes the policy every [eval_every] seconds.
    """
    try:
      while not coord.should_stop():
        self.eval(sess)
        # Sleep until next evaluation cycle
        time.sleep(eval_every)
    except tf.errors.CancelledError:
      return