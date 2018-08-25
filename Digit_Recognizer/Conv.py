import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from sklearn.cross_validation import train_test_split
import math

data_train = pd.read_csv('train.csv').values
data_test = pd.read_csv('test.csv').values

X = data_train[:,1:]
Y = data_train[:,0]

# Split labeled data into train and test
# Final model is trained using complete dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

#X_train.reshape(X_train.shape[0],28,28)
#np.reshape(X_train,(28,28,-1))
#np.reshape(X_test,(28,28,-1))
# Visualize dataset
#print(X[5])
#plt.imshow(X[2].reshape(28,28))
#plt.show()
#print(X[0].shape)
#print(X.shape)
#print(y.shape)
#print(X_train.shape)
#print(X_test.shape)
print(Y_train.shape)

def create_placeholders(n_H0, n_W0, n_y):
	X = tf.placeholder(dtype = tf.float32, shape=(None,n_H0,n_W0,1))
	Y = tf.placeholder(dtype = tf.float32, shape=(None,))

	return X,Y

# X,y = create_placeholders(28,28,10)
# print("X="+str(X))

def initialize_parameters():
	W1 = tf.get_variable("W1", [4,4,1,8],dtype = tf.float32,initializer = tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable("W2", [2,2,8,16],dtype = tf.float32,initializer = tf.contrib.layers.xavier_initializer())
	parameters = {"W1": W1, "W2": W2}

	return parameters

def forward_propagation(X, parameters):
	W1 = parameters["W1"]
	W2 = parameters["W2"]

	Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = "SAME")
	A1 = tf.nn.relu(Z1)
	P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides=[1,8,8,1], padding = "SAME")

	Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1], padding="SAME")
	A2 = tf.nn.relu(Z2)
	P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")

	P2 = tf.contrib.layers.flatten(P2)

	Z3 = tf.contrib.layers.fully_connected(P2, 1, activation_fn=None)

	return Z3

def compute_cost(Z3, Y):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
	return cost

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
    m = X.shape[0]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    #permutation = list(np.random.permutation(m))
    #shuffled_X = X[permutation,:]
    #shuffled_Y = Y[permutation,:]#.reshape((m,1))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = X[k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch_Y = Y[k*mini_batch_size:(k+1)*mini_batch_size,:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = X[num_complete_minibatches*mini_batch_size:num_complete_minibatches*mini_batch_size+(m-mini_batch_size*num_complete_minibatches),:]
        mini_batch_Y = Y[num_complete_minibatches*mini_batch_size:num_complete_minibatches*mini_batch_size+(m-mini_batch_size*num_complete_minibatches),:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def model1(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs = 120, minibatch_size = 64, print_cost = True):
	tf.reset_default_graph()
	seed = 3
	(m, n_H0) = X_train.shape
	n_H0 = np.sqrt(n_H0)
	n_W0 = n_H0
	n_y = 1
	costs = []

	X, Y = create_placeholders(n_H0,n_W0,n_y)

	parameters = initialize_parameters()

	Z3 = forward_propagation(X, parameters)

	cost = compute_cost(Z3,Y)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(num_epochs):
			minibatch_cost = 0
			num_minibatches = int(m/minibatch_size)
			seed = seed + 1
			minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)

			for minibatch in minibatches:

				(minibatch_X, minibatch_Y) = minibatch

				_, temp_cost = sess.run([optimizer,cost], feed_dict= {X:minibatch_X.reshape(64,28,28,1), Y:minibatch_Y.reshape(31500,1)})

				minibatch_cost += temp_cost/num_minibatches

			if print_cost == True and epoch%5==0:
				print("Cost after epoch %i: %f",(epoch, minibatch_cost))
			if print_cost == True and epoch %1==0:
				costs.append(minibatch_cost)

		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations')
		plt.title("learning_rate="+str(learning_rate))
		plt.show()

		predict_op = Z3
		correct_prediction = tf.equal(predict_op, tf.argmax(Y,1))
		
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
		print(accuracy)	
		train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
		test_accuracy = accuracy.eval({X:X_test,Y:Y_test})

		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)

		return train_accuracy, test_accuracy, parameters



def model2(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs = 120, minibatch_size = 64, print_cost = True):
	tf.reset_default_graph()
	seed = 3
	(m, n_H0) = X_train.shape
	n_H0 = np.sqrt(n_H0)
	n_W0 = n_H0
	n_y = 1
	costs = []

	X, Y = create_placeholders(n_H0,n_W0,n_y)

	parameters = initialize_parameters()

	Z3 = forward_propagation(X, parameters)

	cost = compute_cost(Z3,Y)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatch_cost = 0
			num_minibatches = int(m/minibatch_size)
			for _ in range(num_minibatches):
				(minibatch_X, minibatch_Y) = data_train.next_batch(mini_batch_size)
				_, temp_cost = sess.run([optimizer,cost], feed_dict= {X:minibatch_X.reshape(64,28,28,1), Y:minibatch_Y})

				minibatch_cost += temp_cost/num_minibatches

			if print_cost == True and epoch%5==0:
				print("Cost after epoch %i: %f",(epoch, minibatch_cost))
			if print_cost == True and epoch %1==0:
				costs.append(minibatch_cost)

		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations')
		plt.title("learning_rate="+str(learning_rate))
		plt.show()

		predict_op = Z3
		correct_prediction = tf.equal(predict_op, tf.argmax(Y,1))
		
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
		print(accuracy)	
		train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
		test_accuracy = accuracy.eval({X:X_test,Y:Y_test})

		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)

		return train_accuracy, test_accuracy, parameters


def model3(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs = 120, minibatch_size = 64, print_cost = True):
	tf.reset_default_graph()
	seed = 3
	(m, n_H0) = X_train.shape
	n_H0 = np.sqrt(n_H0)
	n_W0 = n_H0
	n_y = 1
	costs = []

	X, Y = create_placeholders(n_H0,n_W0,n_y)

	parameters = initialize_parameters()

	Z3 = forward_propagation(X, parameters)

	cost = compute_cost(Z3,Y)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatch_cost = 0
			num_minibatches = int(m/minibatch_size)
			_, temp_cost = sess.run([optimizer,cost], feed_dict= {X:X_train.reshape(31500,28,28,1), Y:Y_train})

			minibatch_cost += temp_cost/num_minibatches

			if print_cost == True and epoch%5==0:
				print("Cost after epoch %i: %f",(epoch, minibatch_cost))
			if print_cost == True and epoch %1==0:
				costs.append(minibatch_cost)

		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations')
		plt.title("learning_rate="+str(learning_rate))
		plt.show()

		predict_op = Z3
		correct_prediction = tf.equal(predict_op, tf.argmax(Y,1))
		
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
		print(accuracy)	
		train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
		test_accuracy = accuracy.eval({X:X_test,Y:Y_test})

		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)

		return train_accuracy, test_accuracy, parameters









_, _, parameters = model3(X_train, Y_train, X_test, Y_test)