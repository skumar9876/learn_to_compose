"""
@author: dennybritz
Modified for the ComposeNet project by Saurabh Kumar.
"""

import unittest
import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing
from datetime import datetime
import time

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

sys.path.insert(0, '../environments')
sys.path.insert(0, '../rl_agents/utils')
sys.path.insert(0, '../rl_agents/policy_gradient/a4c')
import objects_env

from estimators import ValueEstimator, PolicyEstimator
from policy_eval_assorted import PolicyEval
from policy_eval_visualization import VisualizePolicy
from worker import Worker, make_copy_params_op

tf.flags.DEFINE_string("task", "assorted", "a4c trains all tasks simultaneously")
tf.flags.DEFINE_string("shared_final_layer", False, "propogates gradients from each task to final layer of all agents")
tf.flags.DEFINE_string("model_dir", "experiment_logs/a4c", "Directory to save checkpoints to.")
tf.flags.DEFINE_string("env", "objects_env", "Name of environment.")
tf.flags.DEFINE_string("teleport", False, "Whether or not to allow teleportation of objects across borders.")
tf.flags.DEFINE_integer("t_max", 5, "Number of steps before performing an update.")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 150, "Evaluate the policy every N seconds.")
tf.flags.DEFINE_integer("visualize_every", 120, "Visualize the learned policy every N seconds.")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", None, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS


def make_env(task=None, obj=None, teleport=False):
  if FLAGS.env == 'objects_env':
    goal_arr = [0]*objects_env.World.NUM_OBJECTS
    if task == 'avoid':
      goal_arr[obj] = -1
      # pick another object to be target (does not matter)
      target_obj = 0
      while target_obj == obj:
        target_obj = np.random.randint(0, objects_env.World.NUM_OBJECTS)
      goal_arr[target_obj] = 1
      return objects_env.World(
        task=['avoid'], goal_arr=goal_arr, teleport=teleport)
    elif task == 'target':
      goal_arr[obj] = 1
      # pick another object to be avoid (does not matter)
      avoid_obj = 0
      while avoid_obj == obj:
        avoid_obj = np.random.randint(0, objects_env.World.NUM_OBJECTS)
      goal_arr[avoid_obj] = -1
      return objects_env.World(
        task=['target'], goal_arr=goal_arr, teleport=teleport)

def eval_assorted(evaluators, eval_every, sess, coord):
  '''evaluates the assorted environments every [eval_every] seconds'''
  try:
    while not coord.should_stop():
      for ev in evaluators:
        ev.eval(sess, ev.n_eval)
      # Sleep until next evaluation cycle
      time.sleep(eval_every)

  except tf.errors.CancelledError:
    return


# the number of assorted tasks needed to be learnt
NUM_OBJECTS = objects_env.World.NUM_OBJECTS
envs = []
for obj in range(NUM_OBJECTS):
  envs.append(make_env('avoid', obj=obj, teleport=True))
  envs.append(make_env('target', obj=obj, teleport=True))
VALID_ACTIONS = list(range(envs[0].get_num_actions()))

# Set the number of workers
NUM_WORKERS = multiprocessing.cpu_count()
if FLAGS.parallelism:
  NUM_WORKERS = FLAGS.parallelism

MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints/checkpoints_{}{}".format(FLAGS.task, '/model.ckpt'))
LOG_DIR = os.path.join(MODEL_DIR, "logs/logs_{}".format(FLAGS.task))

# Optionally empty model directory
# if FLAGS.reset:
#  shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
  os.makedirs(CHECKPOINT_DIR)

if not os.path.exists(LOG_DIR):
  os.makedirs(LOG_DIR)

with tf.device("/cpu:0"):
  # Keeps track of the number of updates we've performed
  global_step = tf.Variable(0, name="global_step", trainable=False)

  # different policy and value nets for all tasks
  policy_nets = []
  value_nets = []
  for e in range(len(envs)):
    with tf.variable_scope("global_{}".format(e)) as vs:
      policy_nets.append(PolicyEstimator(
        num_outputs=len(VALID_ACTIONS), state_dims=envs[e].get_state_size()))
      value_nets.append(ValueEstimator(
        reuse=True, state_dims=envs[e].get_state_size()))
  if FLAGS.shared_final_layer:
    # make all final layer weights the same
    initial_copy_ops = []
    for e in range(1, len(envs)):
      initial_copy_ops += make_copy_params_op(
        tf.contrib.slim.get_variables(scope="global_0/policy_net", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
        tf.contrib.slim.get_variables(scope="global_{}/policy_net".format(e), collection=tf.GraphKeys.TRAINABLE_VARIABLES))
      initial_copy_ops += make_copy_params_op(
        tf.contrib.slim.get_variables(scope="global_0/value_net", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
        tf.contrib.slim.get_variables(scope="global_{}/value_net".format(e), collection=tf.GraphKeys.TRAINABLE_VARIABLES))

  # Global step iterator
  global_counter = itertools.count()

  # Create worker graphs
  workers = []
  for worker_id in range(NUM_WORKERS):
    # add a curriculum for avoid task
    env_id = worker_id % len(envs)
    curriculum = None
    if 'avoid' in envs[env_id].task:
      curriculum = [10, 20, 30, 40, 50]
    worker = Worker(
      name="worker_{}".format(worker_id),
      env=envs[env_id],
      env_id = env_id,
      curriculum = curriculum,
      policy_nets=policy_nets,
      value_nets=value_nets,
      shared_final_layer=FLAGS.shared_final_layer,
      global_counter=global_counter,
      discount_factor = 0.99,
      max_global_steps=FLAGS.max_global_steps)
    workers.append(worker)

  saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.01, max_to_keep=10)

  logfile = os.path.join(
    LOG_DIR,
    '{:%Y-%m-%d_%H:%M:%S}.log'.format(datetime.now()))

  # Used to occasionally evaluate the policy and save
  # statistics and checkpoint model.

  eval_envs = []
  for obj in range(NUM_OBJECTS):
    eval_envs.append(make_env('avoid', obj=obj, teleport=True))
    eval_envs.append(make_env('target', obj=obj, teleport=True))
  evals = []
  for i in range(len(eval_envs)):
    curriculum = None
    if 'avoid' in envs[i].task:
      curriculum = [10, 20, 30, 40, 50]
    evals.append(PolicyEval(
      env=eval_envs[i],
      env_id=i,
      curriculum=curriculum,
      policy_net=policy_nets[i],
      saver=saver,
      logfile=logfile,
      checkpoint_path=CHECKPOINT_DIR))

  # Used to occasionally evaluate the policy and record
  # & save videos of it.
  # pv = VisualizePolicy(
  #   env=make_env(FLAGS.task, FLAGS.teleport),
  #   policy_net=policy_net)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  if FLAGS.shared_final_layer:
    # run the initial copy ops to make final layer the same
    sess.run(initial_copy_ops, feed_dict={})
  coord = tf.train.Coordinator()

  '''
  # Load a previous checkpoint if it exists
  latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
  if latest_checkpoint:
    print("Loading model checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)
  '''

  # Start worker threads
  worker_threads = []
  for worker in workers:
    worker_fn = lambda: worker.run(sess, coord, FLAGS.t_max)
    t = threading.Thread(target=worker_fn)
    t.start()
    worker_threads.append(t)

  # Start a thread for policy eval task
  eval_thread = threading.Thread(target=lambda: eval_assorted(evals, FLAGS.eval_every, sess, coord))
  eval_thread.start()

#   visualize_thread = threading.Thread(target=lambda: pv.continuous_eval(FLAGS.visualize_every, sess, coord))
#   visualize_thread.start()

  # Wait for all workers to finish
  coord.join(worker_threads)
