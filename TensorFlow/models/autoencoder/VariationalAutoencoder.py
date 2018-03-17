import tensorflow as tf
from Autoencoder import Autoencoder

class VariationalAutoencoder(Autoencoder):
	def __init__(self, n_input, n_hidden, optimizer=tf.train.AdamOptimizer(), initializer=tf.contrib.layers.xavier_initializer()):
		super().__init__(n_input, n_hidden, optimizer, initializer)
	
	def _build_graph():
		# model
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.z_mean = tf.add(tf.matmul(self.x, self.weights['w1'], self.weights['b1']))
		self.z_log_sigma_sq = tf.add(tf.matmul(self.x, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])

		eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.hidden]), 0, 1, dtype=tf.float32)
		self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
		self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])

	def _define_cost(optimizer):
		reconstr_loss = .5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
		latent_loss = -.5 * tf.reduce_sum(
			1 + self.z_log_sigma_sq
			  - tf.square(self.z_mean)
			  - tf.exp(self.z_log_sigma_sq),
			1
		)

		self.cost = tf.reduce_mean(reconstr_loss, latent_loss)
		self.optimizer = optimizer(self.cost)
	
	def _initialize_weights(self, initializer):
		super(VariationalAutoencoder, self)._initialize_weights(initializer)
		self.weights['log_sigma_w1'] = tf.get_variable('log_sigma_w1', shape=[self.n_input, self.n_hidden], intializer=initializer)
		self.weights['log_sigma_b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
		
