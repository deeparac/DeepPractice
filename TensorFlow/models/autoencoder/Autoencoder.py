import tensorflow as tf

import Autoencoder as ae
class Autoencoder(object):
	def __init__(self, n_input, n_hidden, transfer_funcion=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), initializer=tf.contrib.layers.xavier_initializer()):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.transfer = transfer_function

		self._initialize_weights(initializer)
		self._build_graph()
		self._define_cost(optimizer)
	
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)
	
	def _define_cost(optimizer):
		# cost
		self.cost = .5 * tf.reduce_sum(
			tf.pow(
				tf.subtract(self.reconstruction, self.x), 
				2.0
			)
		)
		self.optimizer = optimizer.minimize(self.cost)	
	
	def _build_graph():
		# model
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.hidden = self.transfer(
			tf.add(
				tf.matmul(self.x, self.weights['w1']),
				self.weights['b1']
			)
		)
		self.reconstruction = tf.add(
			tf.matmul(self.hidden, self.weights['w2']),
			self.weights['b2']
		)

	def _initialize_weights(self, initializer):
		self.weights = {}
		self.weights['w1'] = tf.get_variable('w1', shape=[self.n_input, self.n_hidden], initializer=initializer, dtype=tf.float32)
		self.weights['w2'] = tf.get_variable('w2', shape=[self.n_hidden, self.n_input], initializer=initializer, dtype=tf.float32)
		self.weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)
		self.weights['b2'] = tf.Variable(tf.zeros([self.n_input]), dtype=tf.float32)

	def feed_dict(self, X):
		return {self.x: X)

	def partial_fit(self, X):
		cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict=feed_dict(X))
		return cost
	
	def calc_total_cost(self, X):
		return self.sess.run(self.cost, feed_dict=feed_dict(X))
	
	def transform(self, X):
		return self.sess.run(self.hidden, feed_dict=feed_dict(X))
	
	def generate(self, hidden=None):
		if hidden is None:
			hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
		return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})
	
	def reconstruct(self, X):
		return self.sess.run(self.reconstruction, feed_dict=feed_dict(X))
	
	def getWeights(self):
		return self.sess.run(self.weights['w1'])
	
	def getBiases(self):
		return self.sess.run(self.weights['b1'])

		
