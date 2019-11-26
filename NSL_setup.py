import pandas as pd
import numpy as np
import seaborn as sns
import imblearn
import itertools
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler 
from collections import Counter
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


FEATURE_NUM = 35 # the number of selected features

def preprocessing():

	##################################  LOADING DATA ###################################
	# file paths of training and testing data
	train_file_path = 'E:/downloads/datasets/NSL_KDD/KDDTrain+_20Percent.txt'
	test_file_path = 'E:/downloads/datasets/NSL_KDD/KDDTest+.txt'

	# attributes/features of the data
	datacols = ["duration","protocol_type","service","flag","src_bytes",
	"dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
	"logged_in","num_compromised","root_shell","su_attempted","num_root",
	"num_file_creations","num_shells","num_access_files","num_outbound_cmds",
	"is_host_login","is_guest_login","count","srv_count","serror_rate",
	"srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
	"diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
	"dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
	"dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
	"dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]

	# Load NSL_KDD train dataset
	dfkdd_train = pd.read_table(train_file_path, sep=",", names=datacols) # load data
	dfkdd_train = dfkdd_train.iloc[:,:-1] # removes an unwanted extra field

	# Load NSL_KDD test dataset
	dfkdd_test = pd.read_table(test_file_path, sep=",", names=datacols)
	dfkdd_test = dfkdd_test.iloc[:,:-1]

	# train set dimension
	print('Train set dimension: {} rows, {} columns'.format(dfkdd_train.shape[0], dfkdd_train.shape[1]))
	# test set dimension
	print('Test set dimension: {} rows, {} columns'.format(dfkdd_test.shape[0], dfkdd_test.shape[1]))


	###################################### DATA PREPROCESSING ############################################

	mapping = {'ipsweep': 'Probe','satan': 'Probe','nmap': 'Probe','portsweep': 'Probe','saint': 'Probe','mscan': 'Probe',
		'teardrop': 'DoS','pod': 'DoS','land': 'DoS','back': 'DoS','neptune': 'DoS','smurf': 'DoS','mailbomb': 'DoS',
		'udpstorm': 'DoS','apache2': 'DoS','processtable': 'DoS',
		'perl': 'U2R','loadmodule': 'U2R','rootkit': 'U2R','buffer_overflow': 'U2R','xterm': 'U2R','ps': 'U2R',
		'sqlattack': 'U2R','httptunnel': 'U2R',
		'ftp_write': 'R2L','phf': 'R2L','guess_passwd': 'R2L','warezmaster': 'R2L','warezclient': 'R2L','imap': 'R2L',
		'spy': 'R2L','multihop': 'R2L','named': 'R2L','snmpguess': 'R2L','worm': 'R2L','snmpgetattack': 'R2L',
		'xsnoop': 'R2L','xlock': 'R2L','sendmail': 'R2L',
		'normal': 'Normal'
		}

	# Apply attack class mappings to the dataset
	dfkdd_train['attack_class'] = dfkdd_train['attack'].apply(lambda v: mapping[v])
	dfkdd_test['attack_class'] = dfkdd_test['attack'].apply(lambda v: mapping[v])

	# Drop attack field from both train and test data
	dfkdd_train.drop(['attack'], axis=1, inplace=True)
	dfkdd_test.drop(['attack'], axis=1, inplace=True)

	# 'num_outbound_cmds' field has all 0 values. Hence, it will be removed from both train and test dataset since it is a redundant field.
	dfkdd_train.drop(['num_outbound_cmds'], axis=1, inplace=True)
	dfkdd_test.drop(['num_outbound_cmds'], axis=1, inplace=True)

	# Attack Class Distribution
	attack_class_freq_train = dfkdd_train[['attack_class']].apply(lambda x: x.value_counts())
	attack_class_freq_test = dfkdd_test[['attack_class']].apply(lambda x: x.value_counts())
	attack_class_freq_train['frequency_percent_train'] = round((100 * attack_class_freq_train / attack_class_freq_train.sum()),2)
	attack_class_freq_test['frequency_percent_test'] = round((100 * attack_class_freq_test / attack_class_freq_test.sum()),2)

	attack_class_dist = pd.concat([attack_class_freq_train,attack_class_freq_test], axis=1) 
	#print(attack_class_dist)

	# Attack class bar plot
	plot = attack_class_dist[['frequency_percent_train', 'frequency_percent_test']].plot(kind="bar");
	plot.set_title("Attack Class Distribution", fontsize=20);
	plot.grid(color='lightgray', alpha=0.5);

	# Scaling Numerical Attributes
	scaler = StandardScaler()
	#scaler = MinMaxScaler()
	# extract numerical attributes and scale it to have zero mean and unit variance  
	cols = dfkdd_train.select_dtypes(include=['float64','int64']).columns
	sc_train = scaler.fit_transform(dfkdd_train.select_dtypes(include=['float64','int64']))
	sc_test = scaler.fit_transform(dfkdd_test.select_dtypes(include=['float64','int64']))
	# turn the result back to a dataframe
	sc_traindf = pd.DataFrame(sc_train, columns = cols)
	sc_testdf = pd.DataFrame(sc_test, columns = cols)

	# Encoding of categorical Attributes
	encoder = LabelEncoder()
	# extract categorical attributes from both training and test sets 
	cattrain = dfkdd_train.select_dtypes(include=['object']).copy()
	cattest = dfkdd_test.select_dtypes(include=['object']).copy()
	# encode the categorical attributes
	traincat = cattrain.apply(encoder.fit_transform)
	testcat = cattest.apply(encoder.fit_transform)
	# separate target column from encoded data 
	enctrain = traincat.drop(['attack_class'], axis=1)
	enctest = testcat.drop(['attack_class'], axis=1)

	cat_Ytrain = traincat[['attack_class']].copy()
	cat_Ytest = testcat[['attack_class']].copy()

	# data sampling
	# define columns and extract encoded train set for sampling 
	#####################sc_traindf = dfkdd_train.select_dtypes(include=['float64','int64'])
	refclasscol = pd.concat([sc_traindf, enctrain], axis=1).columns
	refclass = np.concatenate((sc_train, enctrain.values), axis=1)
	X = refclass
	# reshape target column to 1D array shape  
	c, r = cat_Ytest.values.shape
	y_test = cat_Ytest.values.reshape(c,)
	c, r = cat_Ytrain.values.shape
	y = cat_Ytrain.values.reshape(c,)
	# apply the random over-sampling
	ros = RandomOverSampler(random_state=42)
	X_res, y_res = ros.fit_sample(X, y)

	# create test dataframe
	reftest = pd.concat([sc_testdf, testcat], axis=1)
	reftest['attack_class'] = reftest['attack_class'].astype(np.float64)
	reftest['protocol_type'] = reftest['protocol_type'].astype(np.float64)
	reftest['flag'] = reftest['flag'].astype(np.float64)
	reftest['service'] = reftest['service'].astype(np.float64)

	print('Original dataset shape {}'.format(Counter(y)))
	print('Resampled dataset shape {}'.format(Counter(y_res)))

	return X_res, y_res, refclasscol, reftest

def data_partition( X_res, y_res, refclasscol, reftest, selected_features, attackclass):

	######################################### DATA PARTITION  ######################################
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DATA PARTITION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
	attack_name = attackclass[0][0]
	newcol = list(refclasscol)
	newcol.append('attack_class')

	# add a dimension to target
	new_y_res = y_res[:, np.newaxis]

	# create a dataframe from sampled data
	res_arr = np.concatenate((X_res, new_y_res), axis=1)
	res_df = pd.DataFrame(res_arr, columns = newcol) 

	# create two-target classes (normal class and an attack class)  
	classdict = defaultdict(list)
	normalclass = [('Normal', 1.0)]			
	classdict = create_classdict(classdict, res_df, reftest, normalclass, attackclass)


	pretrain = classdict['Normal_'+ attack_name][0]
	pretest = classdict['Normal_'+ attack_name][1]
	grpclass = 'Normal_'+ attack_name

	# finalize data preprocessing for training
	enc = OneHotEncoder()

	Xresdf = pretrain 
	newtest = pretest

	Xresdfnew = Xresdf[selected_features]
	Xresdfnum = Xresdfnew.drop(['service'], axis=1)
	Xresdfcat = Xresdfnew[['service']].copy()

	Xtest_features = newtest[selected_features]
	Xtestdfnum = Xtest_features.drop(['service'], axis=1)
	Xtestcat = Xtest_features[['service']].copy()

	# Fit train data
	enc.fit(Xresdfcat)

	# Transform train and test data
	X_train_1hotenc = enc.transform(Xresdfcat).toarray()
	X_test_1hotenc = enc.transform(Xtestcat).toarray()

	X_train = np.concatenate((Xresdfnum.values, X_train_1hotenc), axis=1)
	X_test = np.concatenate((Xtestdfnum.values, X_test_1hotenc), axis=1) 

	y_train = Xresdf[['attack_class']].copy()
	c, r = y_train.values.shape
	Y_train = y_train.values.reshape(c,)

	y_test = newtest[['attack_class']].copy()
	c, r = y_test.values.shape
	Y_test = y_test.values.reshape(c,)

	# transform the labels to one hot
	Y_train = np.arange(2) == Y_train[:, None].astype(np.float32)
	Y_test = np.arange(2) == Y_test[:, None].astype(np.float32)
	print('X_train', X_train[0],X_train[0].shape)
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DATA PARTITION FINISHED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	return X_train, X_test, Y_train, Y_test


def create_classdict(classdict, res_df, reftest, normalclass, attackclass):
# This function subdivides train and test dataset into two-class attack labels 
	j = normalclass[0][0] # name of normal class
	k = normalclass[0][1] # numerical representer of normal class
	i = attackclass[0][0] # name of abnormal class(DOS, Probe, R2L, U2R)
	v = attackclass[0][1] # numerical representer of normal class [('DoS', 0.0), ('Probe', 2.0), ('R2L', 3.0), ('U2R', 4.0)]
	restrain_set = res_df.loc[(res_df['attack_class'] == k) | (res_df['attack_class'] == v)]
	classdict[j +'_' + i].append(restrain_set)
	# test labels
	reftest_set = reftest.loc[(reftest['attack_class'] == k) | (reftest['attack_class'] == v)]
	classdict[j +'_' + i].append(reftest_set)
	return classdict

def feature_selection(X_res, y_res, xcol, FEATURE_NUM):
	
	###################### feature selections ###########################
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FEATURE SELECTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
	rfc = RandomForestClassifier()
	# fit random forest classifier on the training set
	y_res = y_res.reshape(-1,1) # reshape the lables
	rfc.fit(X_res, y_res)
	# extract important features
	score = np.round(rfc.feature_importances_,3)
	importances = pd.DataFrame({'feature': xcol, 'importance':score})
	importances = importances.sort_values('importance',ascending=False).set_index('feature')
	# plot importances
	plt.rcParams['figure.figsize'] = (11, 4)
	importances.plot.bar()	

	# create the RFE model and select 10 attributes
	rfe = RFE(rfc, FEATURE_NUM)
	rfe = rfe.fit(X_res, y_res)

	# summarize the selection of the attributes
	feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), xcol)]
	selected_features = [v for i, v in feature_map if i==True]

	return selected_features

class NSL_KDD:
	def __init__(self, attack_class):

		X_res, y_res, xcol, reftest = preprocessing() 
		selected_features = feature_selection(X_res, y_res, xcol, FEATURE_NUM)
		train_data, test_data, train_labels, test_labels = data_partition( X_res, y_res, xcol, reftest, selected_features, attack_class)
		train_data = self.data_rerange(train_data)
		test_data = self.data_rerange(test_data)

		self.test_data = test_data[:,0:FEATURE_NUM-2]
		self.test_labels = test_labels
		
		VALIDATION_SIZE = 5000
		
		self.validation_data = train_data[:VALIDATION_SIZE, 0:FEATURE_NUM-2]
		self.validation_labels = train_labels[:VALIDATION_SIZE]
		self.train_data = train_data[VALIDATION_SIZE:, 0:FEATURE_NUM-2]
		self.train_labels = train_labels[VALIDATION_SIZE:]
		self.FEATURE_NUM_FINAL = FEATURE_NUM -2

	def data_rerange(self, data):
		scaler = MinMaxScaler()
		# extract numerical attributes and scale it to have zero mean and unit variance  
		data = scaler.fit_transform(data)-0.5
		return data

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

class NNModel:
    def __init__(self, restore, session=None):
        model = Sequential()
        model.add(Dense(30,input_dim=FEATURE_NUM-2, activation='relu'))
        model.add(Dense(2))
        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)