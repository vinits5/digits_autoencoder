from neural_network import neural_network
import numpy as np
import tensorflow as tf
from neural_network.network_structure.Logger import Logger
import os

network = neural_network.neural_network()
logger = Logger(network.path)

train_data,train_labels,test_data,test_labels = network.data()
EPOCHS = 1000
BATCH_SIZE = 32

def log_write(data):
	with open(os.path.join(network.path,'log.txt'),'a') as file:
		file.write(data+'\n')
	print(data)

def train():
	sess = tf.Session()
	network.create_model()
	network.session_init(sess)
	for epoch in range(EPOCHS):
		log_write('##########%4d##########'%(epoch))
		train_loss,train_accuracy = train_one_epoch(train_data,train_labels,epoch)
		test_accuracy = test_one_epoch(test_data,test_labels)
		logger.log_scalar(tag='Loss per Epoch',value=train_loss,step=epoch)
		# logger.log_scalar(tag='Train Accuracy per Epoch',value=train_accuracy,step=epoch)
		# logger.log_scalar(tag='Test Accuracy per Epoch',value=test_accuracy,step=epoch)
		log_write('Train Loss: '+str(train_loss))
		# log_write('Train Accuracy: '+str(train_accuracy))
		# log_write('Test Accuracy: '+str(test_accuracy))
		if (epoch%100)==0:
			network.save_weights(epoch)

def train_one_epoch(train_data,train_labels,epoch_id):
	file_size = train_data.shape[0]
	batch_idxs = file_size//BATCH_SIZE
	total_loss = 0
	total_accuracy = 0
	total_seen = 0.0
	for idx in range(batch_idxs):
		start_idx = idx*BATCH_SIZE
		end_idx = (idx+1)*BATCH_SIZE

		current_data = train_data[start_idx:end_idx,:]
		current_labels = train_labels[start_idx:end_idx,:].flatten()

		pred = network.forward(ip=current_data,is_training=True,batch_size=BATCH_SIZE)
		loss = network.backward(ip=current_data,y=current_data,batch_size=BATCH_SIZE,is_training=True)

		pred_val = np.argmax(pred,axis=1)
		accuracy = np.sum(pred_val==current_labels)
		
		total_accuracy += accuracy
		total_seen += BATCH_SIZE
		total_loss += loss
		# logger.log_scalar(tag='Loss per batch',value = loss,step=(epoch_id+idx))
	return (total_loss/(batch_idxs*1.0)),(total_accuracy/total_seen)

def test_one_epoch(test_data,test_labels):
	test_labels = test_labels.flatten()	
	pred = network.forward(ip=test_data,is_training=False,batch_size=100)
	
	pred_val = np.argmax(pred,1)
	accuracy = np.sum(pred_val==test_labels)
	accuracy = accuracy/(test_labels.shape[0]*1.0)
	return accuracy

if __name__=='__main__':
	train()