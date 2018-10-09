import tensorflow as tf 
from network_structure.network_structure import network_structure
import numpy as np
import os
import datetime
import shutil
import sys
import csv

class neural_network():
	# Initialize the path for storing data.
	def __init__(self):
		# Class for neural network structure.
		self.ns = network_structure()
		now = datetime.datetime.now()
		path = os.getcwd()
		try:
			os.mkdir('log_data')
		except:
			pass
		self.path = os.path.join(path,'log_data/',now.strftime("%Y-%m-%d-%H-%M-%S"))
		# os.mkdir(self.path)

	# Initialize the variables of Neural Network.
	def session_init(self,sess):
		self.sess = sess
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep = 100)

	# Define the network structure.
	def create_model(self):
		self.ns.structure()

	# Forward pass of neural network.
	def forward(self,ip,is_training,batch_size):
		op = self.sess.run([self.ns.output],feed_dict={self.ns.x:ip,self.ns.is_training:is_training})
		return np.asarray(op).reshape((batch_size,784))

	# Backpropagation of neural network.
	def backward(self,ip,y,batch_size,is_training):
		loss,_ = self.sess.run([self.ns.loss,self.ns.updateModel],feed_dict={self.ns.x:ip,self.ns.y:y,self.ns.batch_size:batch_size,self.ns.is_training:is_training})
		return loss

	# Store weights for further use.
	def save_weights(self,episode):
		path_w = os.path.join(self.path,'weights')
		try:
			os.chdir(path_w)
		except:
			os.mkdir(path_w)
			os.chdir(path_w)
		path_w = os.path.join(path_w,'{}.ckpt'.format(episode))
		self.saver.save(self.sess,path_w)

	def load_weights(self,weights):
		self.saver.restore(self.sess,weights)

	# Store network structure in logs.
	def save_network_structure(self):
		curr_dir = os.getcwd()
		src_path = os.path.join(curr_dir,'neural_network','network_structure','network_structure.py')
		target_path = os.path.join(self.path,'network_structure.py')
		shutil.copy(src_path,target_path)

	def print_data(self,text,step,data):
		text = "\r"+text+" %d: %f"
		sys.stdout.write(text%(step,data))
		sys.stdout.flush()

	def batch_size(self,batch):
		self.sess.run(self.ns.batch_size,feed_dict={self.ns.batch_size:batch})

	def data(self):
		train_data,train_labels = [],[]
		with open(os.path.join(os.getcwd(),'data/train.csv'),'r') as csvfile:
			csvreader = csv.reader(csvfile)
			csvreader.next()
			for row in csvreader:
				train_labels.append(int(row.pop(0)))
				row = [int(i) for i in row]
				train_data.append(row)

		train_data = np.asarray(train_data)
		test_data = train_data[train_data.shape[0]-100:train_data.shape[0],:]
		train_data = train_data[0:train_data.shape[0]-100,:]
		train_labels = np.asarray(train_labels).reshape(len(train_labels),1)
		test_labels = train_labels[train_labels.shape[0]-100:train_labels.shape[0],:]
		train_labels = train_labels[0:train_labels.shape[0]-100,:]
		return train_data,train_labels,test_data,test_labels
