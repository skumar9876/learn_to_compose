import tensorflow as tf

'''
Network definition functions.
'''
def weight_variable(shape, name):
    # initial = tf.truncated_normal(shape, stddev=0.01)
    # return tf.Variable(initial)
    return tf.get_variable(name, shape=shape,
    	initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class Transition(object):

	def __init__(self, state, action, reward, next_state, terminal):
		self.state = state
		self.action = action
		self.reward = reward
		self.next_state = next_state
		self.terminal = terminal