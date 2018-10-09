from neural_network import neural_network
import numpy as np
import tensorflow as tf
from neural_network.network_structure.Logger import Logger
import os
import matplotlib.pyplot as plt
import sys

network = neural_network.neural_network()

train_data,train_labels,test_data,test_labels = network.data()

data = int(sys.argv[1])
image = train_data[data,:].reshape((28,28))

network.create_model()
sess = tf.Session()
network.session_init(sess)

is_training = False
network.load_weights('log_data/2018-10-09-09-21-28/weights/900.ckpt')
BATCH_SIZE = 1

pred = network.forward(train_data[data,:].reshape((1,784)),is_training,BATCH_SIZE)
pred = pred.reshape((28,28))
pred_val = np.argmax(pred)
# print pred_val

plt.imshow(image,cmap='gray')
plt.title('Number in Image: '+str(train_labels[data,0])+' & Predicted Number: '+str(pred_val))
plt.figure()
plt.imshow(pred,cmap='gray')
plt.show()