import tensorflow as tf 


class network_structure():
	# Use it to initialize weights.
	def weights(self,x,y):
		weights_dict = {'weights':tf.Variable(tf.random_normal([x,y])),'biases':tf.Variable(tf.random_normal([y]))}
		return weights_dict

	# Define the complete neural network here.
	def structure(self):
		self.x = tf.placeholder(tf.float32,shape=(None,784))
		self.y = tf.placeholder(tf.float32,shape=(None,784))
		self.is_training = tf.placeholder(tf.bool)

		self.batch_size = tf.placeholder(tf.float32,shape=None)

		self.l4 = tf.contrib.layers.fully_connected(self.x,32)

		self.l5 = tf.contrib.layers.fully_connected(self.l4,32)

		self.output = tf.contrib.layers.fully_connected(self.l5,784,activation_fn=None)
		
		self.loss = tf.square(tf.subtract(self.output,self.y))
		self.loss = tf.reduce_mean(self.loss)
		self.loss = tf.divide(self.loss,self.batch_size)
		self.trainer = tf.train.AdamOptimizer(learning_rate = 0.00001)
		self.updateModel = self.trainer.minimize(self.loss)