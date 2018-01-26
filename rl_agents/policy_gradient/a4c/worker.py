"""
@author: dennybritz
Modified for the ComposeNet project by Saurabh Kumar.
"""

import gym
import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

from estimators import ValueEstimator, PolicyEstimator

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def make_copy_params_op(v1_list, v2_list):
  """
  Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
  The ordering of the variables in the lists must be identical.
  """
  v1_list = list(sorted(v1_list, key=lambda v: v.name))
  v2_list = list(sorted(v2_list, key=lambda v: v.name))

  update_ops = []
  for v1, v2 in zip(v1_list, v2_list):
    op = v2.assign(v1)
    update_ops.append(op)

  return update_ops

class Worker(object):
  """
  An A4C worker thread. Runs episodes locally and updates global shared value and policy nets
  for the right environment.
  Args:
    name: A unique name for this worker.
    env: The environment used by this worker.
    env_id: Environment id and which net to upate
    policy_nets: All globally shared policy networks.
    value_nets: All globally shared value networks.
    global_counter: Iterator that holds the global step.
    discount_factor: Reward discount factor.
    max_global_steps: If set, stop coordinator when global_counter > max_global_steps.
  """
  def __init__(
      self, name, env, env_id, curriculum, policy_nets, value_nets,
      shared_final_layer, global_counter, discount_factor=0.99,
      summary_writer=None, max_global_steps=None):
    self.name = name
    self.discount_factor = discount_factor
    self.max_global_steps = max_global_steps
    self.global_step = tf.contrib.framework.get_global_step()
    self.global_policy_nets = policy_nets
    self.global_value_nets = value_nets
    self.global_counter = global_counter
    self.local_counter = itertools.count()
    self.env = env
    self.env_id = env_id
    self.curriculum = curriculum
    self.shared_final_layer = shared_final_layer

    # Create local policy/value nets that are not updated asynchronously
    with tf.variable_scope(name):
      self.policy_net = PolicyEstimator(policy_nets[0].num_outputs, state_dims=env.get_state_size())
      self.value_net = ValueEstimator(reuse=True, state_dims=env.get_state_size())

    # Op to copy params from global policy/valuenets
    self.copy_params_op = make_copy_params_op(
      tf.contrib.slim.get_variables(scope="global_{}".format(env_id), collection=tf.GraphKeys.TRAINABLE_VARIABLES),
      tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES))

    self.vnet_train_op = self.make_train_op(
      self.value_net, self.global_value_nets)
    self.pnet_train_op = self.make_train_op(
      self.policy_net, self.global_policy_nets)
    if self.shared_final_layer:
      # create ops to train the final layers of all other agents
      self.policy_layer_train_ops = self.make_final_layer_train_ops(
        self.policy_net, self.global_policy_nets, 'policy')
      self.value_layer_train_ops = self.make_final_layer_train_ops(
        self.value_net, self.global_value_nets, 'value')

    self.state = None
    self.epochs = 0

  def make_train_op(self, local_estimator, global_estimators):
    """
    Creates an op that applies local estimator gradients
    to the corresponding global estimator (the one with the
    same environment id as the local estimator).
    """
    local_grads, _ = zip(*local_estimator.grads_and_vars)
    # Clip gradients
    local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
    _, global_vars = zip(*global_estimators[self.env_id].grads_and_vars)
    local_global_grads_and_vars = list(zip(local_grads, global_vars))
    return global_estimators[self.env_id].optimizer.apply_gradients(
      local_global_grads_and_vars,
      global_step=tf.contrib.framework.get_global_step())

  def make_final_layer_train_ops(
      self, local_estimator, global_estimators, mode):
    """
    updates the policy layer of all global estimators with local gradients
    """
    final_layer_ops = []
    for i in range(len(global_estimators)):
      if i != self.env_id:
        _, global_vars = zip(*global_estimators[i].grads_and_vars)
        local_global_grads_and_vars = []
        for v, var in enumerate(global_vars):
          if mode+'_net' in var.name:
            # have to search for this variable in local grads
            truncated_name = var.name.replace('global_{}/'.format(i),'')
            for local_grad, local_var in local_estimator.grads_and_vars:
                if truncated_name in local_var.name:
                    local_global_grads_and_vars.append((local_grad, var))
        final_layer_ops.append(global_estimators[self.env_id].optimizer.apply_gradients(
          local_global_grads_and_vars,
          global_step=None))
    return final_layer_ops

  def run(self, sess, coord, t_max):
    with sess.as_default(), sess.graph.as_default():
      # Initial state
      self.state = self.env.reset()
      try:
        while not coord.should_stop():
          # Copy Parameters from the global networks
          sess.run(self.copy_params_op)

          # Collect some experience
          transitions, local_t, global_t = self.run_n_steps(t_max, sess)

          if self.max_global_steps is not None and global_t >= self.max_global_steps:
            tf.logging.info("Reached global step {}. Stopping.".format(global_t))
            coord.request_stop()
            return

          # Update the global networks
          self.update(transitions, sess)

      except tf.errors.CancelledError:
        return

  def _policy_net_predict(self, state, sess):
    feed_dict = { self.policy_net.states: [state] }
    preds = sess.run(self.policy_net.predictions, feed_dict)
    return preds["probs"][0]

  def _value_net_predict(self, state, sess):
    feed_dict = { self.value_net.states: [state] }
    preds = sess.run(self.value_net.predictions, feed_dict)
    return preds["logits"][0]

  def run_n_steps(self, n, sess):
    transitions = []
    for _ in range(n):
      # Take a step
      action_probs = self._policy_net_predict(self.state, sess)
      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      next_state, reward, done = self.env.step(action)

      # Store transition
      transitions.append(Transition(
        state=self.state, action=action, reward=reward, next_state=next_state, done=done))

      # Increase local and global counters
      local_t = next(self.local_counter)
      global_t = next(self.global_counter)

      if local_t % 100 == 0:
        tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))

      if done:
        if global_t > (self.epochs + 1) * 300000:
          self.epochs += 1
        if self.curriculum:
          if self.epochs < len(self.curriculum):
            self.state = self.env.reset(max_steps=self.curriculum[self.epochs])
          else:
            self.state = self.env.reset(max_steps=self.curriculum[-1])
        else:
          self.state = self.env.reset()
        break
      else:
        self.state = next_state
    return transitions, local_t, global_t

  def update(self, transitions, sess):
    """
    Updates global policy and value networks based on collected experience
    Args:
      transitions: A list of experience transitions
      sess: A Tensorflow session
    """

    # If we episode was not done we bootstrap the value from the last state
    reward = 0.0
    if not transitions[-1].done:
      reward = self._value_net_predict(transitions[-1].next_state, sess)

    # Accumulate minibatch exmaples
    states = []
    policy_targets = []
    value_targets = []
    actions = []

    for transition in transitions[::-1]:
      reward = transition.reward + self.discount_factor * reward
      policy_target = (reward - self._value_net_predict(transition.state, sess))
      # Accumulate updates
      states.append(transition.state)
      actions.append(transition.action)
      policy_targets.append(policy_target)
      value_targets.append(reward)

    feed_dict = {
      self.policy_net.states: np.array(states),
      self.policy_net.targets: policy_targets,
      self.policy_net.actions: actions,
      self.value_net.states: np.array(states),
      self.value_net.targets: value_targets,
    }

    # Train the global estimators using local gradients
    global_step, pnet_loss, vnet_loss, _, _ = sess.run([
      self.global_step,
      self.policy_net.loss,
      self.value_net.loss,
      self.pnet_train_op,
      self.vnet_train_op
    ], feed_dict)

    # now run all the policy layer training ops
    if self.shared_final_layer:
      sess.run(self.policy_layer_train_ops, feed_dict)
      sess.run(self.value_layer_train_ops, feed_dict)

    return pnet_loss, vnet_loss, _, _
