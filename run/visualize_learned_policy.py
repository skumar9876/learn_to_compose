"""
Visualizes the policy the agent has learned by allowing the agent to 
interact with the environment for several episodes using the trained
policy model.

@author: Saurabh Kumar
"""
# -*- noplot -*-
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import tensorflow as tf
import sys

sys.path.insert(0, '../environments')

from objects_env import World

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Agent Evaluation', artist='Matplotlib',
                comment='Evaluation!')
writer = FFMpegWriter(fps=3, metadata=metadata)

def make_plot(curr_map, agent, obj1, obj2, obj3, writer):
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

def prob_sample(probs):
	probs = probs[0]
	return np.random.choice(np.arange(len(probs)), p=probs)


with tf.Session() as sess_test:
	new_saver = tf.train.import_meta_graph('experiment_logs/a3c/checkpoints/checkpoints_target/model.ckpt.meta')
	new_saver.restore(sess_test, 'experiment_logs/a3c/checkpoints/checkpoints_target/model.ckpt')
	sess_test.run(tf.global_variables_initializer()) 
	# ^ This is wrong! All variables should already be set to some particular values in the loaded model!
	print tf.get_collection('state')

	action_probs = tf.get_collection('action_probs')[-1]
	state = tf.get_collection('state')[-1]

	fig = plt.figure()
	agent, = plt.plot([], [], 'ko', markersize=20)
	obj1, = plt.plot([], [], 'b*', markersize=20)
	obj2, = plt.plot([], [], 'r*', markersize=20)
	obj3, = plt.plot([], [], 'y*', markersize=20)

	objects = [obj1, obj2, obj3]

	plt.xlim(-1, 5)
	plt.ylim(-1, 5)

	with writer.saving(fig, "writer_test.mp4", 100):

		# Test the trained network for 10 episodes.
		total_reward = []
		episode_entropies = []
		for i in xrange(10):

			env = World()
			curr_map = env.reset()
			make_plot(curr_map, agent, obj1, obj2, obj3, writer)

			episode_reward = 0
			done = False
			while not done:
				feed_dict={state: [curr_map]}

				pred_action = sess_test.run(action_probs, feed_dict=feed_dict)
				action_chosen = prob_sample(pred_action)

				curr_map, reward, done = env.step(action_chosen)
				make_plot(curr_map, agent, obj1, obj2, obj3, writer)
				
				episode_reward += reward

				# curr_action_entropy = sess_test.run(action_entropy, feed_dict=feed_dict)
				# episode_entropies.append(curr_action_entropy)

			total_reward.append(episode_reward)

	# episode_entropies = np.array(episode_entropies)
	# print np.mean(episode_entropies)

	total_reward = np.array(total_reward)
	print np.mean(total_reward)
