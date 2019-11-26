import pickle
import matplotlib.pyplot as plt
import numpy as np

ETA = 100 # step size for gradient descent
LAMBDA = 0.15

def predict(net, n):
	# Get the data from the test set
	x = X_test[n]

	# Print the prediction of the network
	print('Network output: \n' + str(np.round(net.predict_proba(x), 2)) + '\n')
	print('Network prediction: ' + str(np.argmax(net.predict_proba(x))) + '\n')
	

def sigmoid(z):
	"""The sigmoid function."""
	return 1.0/(1.0+np.exp(-z))
																																												
def sigmoid_prime(z):
	"""Derivative of the sigmoid function."""
	return sigmoid(z)*(1-sigmoid(z))

def input_derivative(net, x, y):
	""" Calculate derivatives wrt the inputs"""
	nabla_b = [np.zeros(b.shape) for b in net.intercepts_]
	nabla_w = [np.zeros(w.shape) for w in net.coefs_]
	
	# feedforward
	activation = x
	activations = [x] # list to store all the activations, layer by layer
	zs = [] # list to store all the z vectors, layer by layer
	for b, w in zip(net.intercepts_, net.coefs_):
		z = np.dot(activation,w)+b
		zs.append(z)
		activation = sigmoid(z)
		activations.append(activation)
		
	# backward pass
	delta = (activations[-1]- y) * sigmoid_prime(zs[-1])
	nabla_b[-1] = delta
	nabla_w[-1] = np.dot(activations[-2].transpose(), delta)

	for l in range(2, net.n_layers_):
		z = zs[-l]
		sp = sigmoid_prime(z)
		delta = np.dot(delta,net.coefs_[-l+1].transpose()) * sp
		nabla_b[-l] = delta
		nabla_w[-l] = np.dot(activations[-l-1].transpose(),delta)
		
	# Return derivatives WRT to input
	return delta.dot(net.coefs_[0].T)

def sneaky_adversarial(net, n, x_target, steps, eta, lam=LAMBDA):
	"""
	net : network object
		neural network instance to use
	n : integer
		our goal label (just an int, the function transforms it into a one-hot vector)
	x_target : numpy vector
		our goal image for the adversarial example
	steps : integer
		number of steps for gradient descent
	eta : float
		step size for gradient descent
	lam : float
		lambda, our regularization parameter. Default is .05
	"""
	
	# Set the goal output
	goal = n  # 0 benign; 1 molicious
	feature_num_final = x_target.shape[0]

	# Create a random image to initialize gradient descent with
	x = np.random.normal(.5, .3, (1,feature_num_final))

	# Gradient descent on the input
	for i in range(steps):
		# Calculate the derivative
		d = input_derivative(net,x,goal)
		
		# The GD update on x, with an added penalty to the cost function
		# ONLY CHANGE IS RIGHT HERE!!!
		x -= eta * (d + lam * (x - x_target))

	return x

# Wrapper function
def sneaky_generate(n, m, net, X_test, Y_test):
	"""
	n: int 0-9, the target number to match
	m: index of example image to use (from the test set)
	"""
	
	# Find random instance of m in test set
	idx = np.random.randint(0,Y_test.shape[0]-1)
	while Y_test[idx] != m:
		idx += 1
	
	print('\n the original input is:')
	print(X_test[idx])
	print('True label: ', Y_test[idx])

	# Hardcode the parameters for the wrapper function
	a = sneaky_adversarial(net, n, X_test[idx], ETA, LAMBDA)
	print('\n the adversarial example (based on the above original input) is: ')
	print(a)
	x = np.round(net.predict_proba(a), 2)

	print('Network Output: ' ,x, 'Class: ', np.argmax(x))
	
	return a




