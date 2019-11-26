# main test
import network.network as Network
import network.mnist_loader as mnist_loader
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('network/trained_network.pkl', 'rb') as f:
	u = pickle._Unpickler( f )
	u.encoding = 'latin1'
	net = u.load()

# PYTHON 3 WORK AROUND (uncomment this
# and comment the above if using python 3)
#with open('network/trained_network.pkl', 'rb') as f:
#    u = pickle._Unpickler(f)
#    u.encoding = 'latin1'
#    net = u.load()
    
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
test_data = list(test_data)
validation_data = list(validation_data)
test_data = list(test_data)


def predict(n):
    # Get the data from the test set
    x = test_data[n][0]

    # Print the prediction of the network
    print('Network output: \n' + str(np.round(net.feedforward(x), 2)) + '\n')
    print('Network prediction: ' + str(np.argmax(net.feedforward(x))) + '\n')
    print('Actual image: ')
    
    # Draw the image
    plt.imshow(x.reshape((28,28)), cmap='Greys')

# Replace the argument with any number between 0 and 9999
predict(8384)
