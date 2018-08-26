import numpy as np
import tensorflow as tf 
from tensorflow.python.framework import ops
#from tf_utils import random_mini_batches,predict
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
import math


#####---------------Data Preprocessing-------------------#####
data_train = pd.read_csv("train.csv").values
data_test = pd.read_csv("test.csv").values

X = data_train[1:,1:6]
Y = data_train[1:,6]

#print(X[:,4])
#print(Y)

##### Encoding color column #####
le1 = LabelEncoder()
le1.fit(X[:,4])
color = le1.transform(X[:,4])
color = color/5
#print(color)
#print(X[0:5,:])

##### Replacing color column in X with encodings ####
X[:,4] = color
#print(X[0:5,:])

##### Encoding Labels #####
le2 = LabelEncoder()
le2.fit(Y)
Y_enc = le2.transform(Y)	
#print(Y_enc.shape)
#print(X.shape)
#Y = tf.transpose(tf.one_hot(Y_enc,depth=3))
#X = np.transpose(X)
#print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y_enc)
X_train = X_train.T
X_test = X_test.T 
Y_train = tf.Session().run(tf.transpose(tf.one_hot(Y_train,depth=3)))   # tf.Session().run() returns the tensor as a numpy array - Necessary because while running optimizer 
Y_test = tf.Session().run(tf.transpose(tf.one_hot(Y_test,depth=3)))		# a tensor cannot be passed in feeddict

#####-----------------------------------------------------#####


#####----------------Neural Network Architecture---------------####

from numpy import random
def get_mini_batches(X, y, batch_size):
    random_idxs = random.choice(len(X[1]), len(X[1]), replace=False)
    X_shuffled = X[random_idxs,:]
    y_shuffled = y[random_idxs]
    mini_batches = [(X_shuffled[i:i+batch_size,:], y_shuffled[i:i+batch_size]) for
                   i in range(0, len(X[1]), batch_size)]
    return mini_batches

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
        # mini_batch_X = X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        # mini_batch_Y = Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size:m]
        # mini_batch_X = X[:,num_complete_minibatches*mini_batch_size:m]
        # mini_batch_Y = Y[:,num_complete_minibatches*mini_batch_size:m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def create_placeholders(n_x,n_y):
	X = tf.placeholder(dtype = tf.float32, shape=(n_x,None))
	Y = tf.placeholder(dtype = tf.float32, shape=(n_y,None))

	return X,Y 

def initialize_parameters():
	tf.set_random_seed(1)
	W1 = tf.get_variable("W1", [10,5], initializer = tf.contrib.layers.xavier_initializer(seed=1))
	b1 = tf.get_variable("b1", [10,1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [15,10], initializer = tf.contrib.layers.xavier_initializer(seed=1))
	b2 = tf.get_variable("b2", [15,1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [5,15], initializer = tf.contrib.layers.xavier_initializer(seed=1))
	b3 = tf.get_variable("b3", [5,1], initializer = tf.zeros_initializer())
	W4 = tf.get_variable("W4", [3,5], initializer = tf.contrib.layers.xavier_initializer(seed=1))
	b4 = tf.get_variable("b4", [3,1], initializer = tf.zeros_initializer())

	parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2, "W3":W3, "b3":b3, "W4":W4, "b4":b4}
	return parameters

def forward_propagation(X, parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W3 = parameters["W3"]
	b3 = parameters["b3"]
	W4 = parameters["W4"]
	b4 = parameters["b4"]

	regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
	tf.contrib.layers.apply_regularization(regularizer=regularizer,weights_list=[W1,W2,W3,W4])
	Z1 = tf.add(tf.matmul(W1,X),b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2,A1),b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3,A2),b3)
	A3 = tf.nn.relu(Z3)
	Z4 = tf.add(tf.matmul(W4,A3),b4)

	return Z4

def compute_cost(Z4,Y):
	print(Z4)
	print(Y)
	logits = tf.transpose(Z4)
	labels = tf.transpose(Y)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))
	return cost

def model(X_train,Y_train,X_test,Y_test, learning_rate=0.0005, num_epochs = 10000, minibatch_size=32, print_cost=True):
	ops.reset_default_graph()
	tf.set_random_seed(1)
	seed = 3
	(n_x, m) = X_train.shape
	n_y = Y_train.shape[0]
	costs = []

	X,Y = create_placeholders(n_x,n_y)

	parameters = initialize_parameters()

	Z4 = forward_propagation(X, parameters)

	cost = compute_cost(Z4,Y)

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(num_epochs):
			epoch_cost = 0
			num_minibatches = int(m/minibatch_size)
			seed = seed+1
			minibatches = random_mini_batches(X_train,Y_train, minibatch_size)

			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch

				_, minibatch_cost = sess.run([optimizer,cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})
				epoch_cost += minibatch_cost/num_minibatches
			

			if print_cost == True and epoch%100 ==0:
				print("Cost after epoch %d: %f" % (epoch,epoch_cost))
			if print_cost == True and epoch%5==0:
				costs.append(epoch_cost)

		parameters = sess.run(parameters)

		correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print("Train Accuracy: ", accuracy.eval({X:X_train,Y:Y_train}))
		print("Test Accuracy: ", accuracy.eval({X:X_test, Y:Y_test}))
		return parameters

parameters = model(X_train,Y_train,X_test,Y_test)
