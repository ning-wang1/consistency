import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import imblearn
from sklearn.semi_supervised import LabelSpreading

# user define library
import environment
from NSL_setup import NSL_KDD
import classifier as clf
import adver

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def generate_data(test_data, test_labels, original = 1, target = 0, samples=3):
	inputs = []
	targets = []
	original_labels =[]

	for i in range(samples):
		idx = np.random.randint(0,test_labels.shape[0]-1)
		while np.argmax(test_labels[idx]) != original:
			idx += 1
		inputs.append(test_data[idx])
		targets.append(np.eye(data.test_labels.shape[1])[target])
		original_labels.append(test_labels[idx])

	inputs = np.array(inputs)
	targets = np.array(targets)
	original_labels = np.array(original_labels)
	return inputs, targets, original_labels


if __name__ == "__main__":
	# Possible seletion
	classifiers = {'KNN', 'LGR', 'BNB', 'DTC'} # ML models
	attack_list = [('DoS', 0.0), ('Probe', 2.0), ('R2L', 3.0), ('U2R', 4.0)] # attack classes


	# experiment setting (from possible selection)
	attackclass = [('DoS', 0.0)]

	# data preprocessing and data partition
	data = NSL_KDD(attackclass)

	# Model training and model evaluation
	#model = clf.classifier(classifier_name, train_data, train_labels)
	#clf.evaluate(classifier_name, model, test_data, test_labels)

	# to deal with memory error (use small subset of data)
	train_data = data.train_data[0:5000,:]
	test_data = data.test_data[0:200,:]
	train_labels_one_hot = data.train_labels[0:5000]
	test_labels_one_hot = data.test_labels[0:200]
	train_labels = np.argmax(train_labels_one_hot,1)
	test_labels = np.argmax(test_labels_one_hot,1)

	x_all = np.concatenate((train_data, test_data)) # concatenate the train and test data (for structure exploitation)
	test_labels_none = -1*np.ones([test_labels.shape[0],]) # the label of the test_data is set to -1
	y_all = np.concatenate((train_labels,test_labels_none)) # concatenate the train labels and -1 test labels

	consist_model = LabelSpreading(gamma=4, max_iter=60) 
	consist_model.fit(x_all, y_all)
	clf.evaluate_sub('consistency model', test_labels, consist_model.predict(test_data))

	# lgr_model = clf.classifier('LGR', train_data, train_labels)
	# clf.evaluate('LGR', lgr_model, test_data, test_labels)

	# knn_model = clf.classifier('KNN', train_data, train_labels)
	# clf.evaluate('KNN', knn_model, test_data, test_labels)

	# svm_model = clf.classifier('SVM', train_data, train_labels)
	# clf.evaluate('SVM', svm_model, test_data, test_labels)

	# dtc_model = clf.classifier('DTC', train_data, train_labels)
	# clf.evaluate('DTC', dtc_model, test_data, test_labels)

	model_to_attack = clf.classifier('MLP', train_data, train_labels)
	adv = adver.sneaky_generate(0,1,model_to_attack, test_data, test_labels) # 0 is the target, 1 is the origin label

	# consistency model
	print('the consistency model classify the adversarial example as: ', consist_model.predict(adv))
	# print('the LGR model classify the adversarial example as: ', lgr_model.predict(adv))
	# print('the KNN model classify the adversarial example as: ', knn_model.predict(adv))
	# print('the SVM model classify the adversarial example as: ', svm_model.predict(adv))
	# print('the DTC model classify the adversarial example as: ', dtc_model.predict(adv))

	from attack_l2 import CarliniL2
	import tensorflow as tf
	from NSL_setup import NNModel
	import time

	with tf.Session() as sess:
		model_to_attack = NNModel("models/nsl_kdd",sess)
		attack = CarliniL2(sess, model_to_attack, batch_size=1, max_iterations=1000, confidence=0)
		# #attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
		# #				   largest_const=15)
		inputs, targets, original_labels = generate_data(test_data,test_labels_one_hot,samples = 1)

		print('shape>>>>>>>>>>>>>>>>>>')
		print(inputs.shape)
		print(targets.shape)
		print(original_labels.shape)

		timestart = time.time()
		adv = attack.attack(inputs, targets)
		timeend = time.time()

		# print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

		# for i in range(len(adv)):		

		# 	print("Valid:")
		# 	print('Original Labels: ', original_labels[i], ' Class: ', np.argmax(original_labels))
		# 	#show(inputs[i])			

		# 	print("Adversarial:")
		# 	#show(adv[i])
		# 	outputs = softmax_exp(model.model.predict(adv[i:i+1])[0])
		# 	print("Classification before softmax: ", outputs)
		# 	print("After softmax: ", outputs, ' Class: ', np.argmax(outputs))

		# 	print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)



	