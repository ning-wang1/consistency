import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import imblearn

# user define library
import environment
import preprocessing as prep
import classifier as cf


models = {'KNN', 'LGR', 'BNB', 'DTC'}
N_FEATURE = 15 # the number of selected features
classifier_name = 'KNN'

# data preprocessing and data partition
X_res, y_res, xcol, reftest = prep.preprocessing() 
selected_features = cf.feature_selection(X_res, y_res, xcol, N_FEATURE)
X_train, X_test, Y_train, Y_test = prep.data_partition( X_res, y_res, xcol, reftest, selected_features)

# Model training and model evaluation
model = cf.classifier(classifier_name, X_train, Y_train)
cf.evaluate(classifier_name, model, X_test, Y_test)


	