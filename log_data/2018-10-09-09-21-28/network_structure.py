import tensorflow as tf

class network_structure():
	# Use it to initialize weights.
	def weights(self,x,y):
		weights_dict = {'weights':tf.Variable(tf.random_normal([x,y])),'biases':tf.Variable(tf.random_normal([y]))}
		return weights_dict

	# Define the complete neural network here.
	def structure(self):
		self.x = tf.placeholder(tf.float32,shape=(None,784))
		self.x = tf.reshape(self.x,[-1,28,28,1])
		self.y = tf.placeholder(tf.float32,shape=(None,784))
		self.y = tf.reshape(self.y,[-1,28,28,1])
		self.is_training = tf.placeholder(tf.bool)

		self.batch_size = tf.placeholder(tf.float32,shape=None)

		self.l1 = tf.layers.conv2d(inputs=self.x,filters=32,kernel_size=4,strides=2,padding="VALID")
		self.l2 = tf.layers.conv2d(inputs=self.l1,filters=32,kernel_size=3,strides=2,padding="VALID")
		self.l3 = tf.layers.conv2d(inputs=self.l2,filters=32,kernel_size=1,strides=1,padding="VALID")
		self.l4 = tf.layers.conv2d_transpose(inputs=self.l3,filters=32,kernel_size=3,strides=2,padding="VALID")
		self.output = tf.layers.conv2d_transpose(inputs=self.l4,filters=1,kernel_size=4,strides=2,padding="VALID")
		
		self.loss = tf.square(tf.subtract(self.output,self.y))
		self.loss = tf.reduce_mean(self.loss)
		self.loss = tf.divide(self.loss,self.batch_size)
		self.trainer = tf.train.AdamOptimizer(learning_rate = 0.00001)
		self.updateModel = self.trainer.minimize(self.loss)
