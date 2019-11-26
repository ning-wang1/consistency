from sklearn.svm import SVC 
from sklearn.naive_bayes import BernoulliNB 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier as MLP

from sklearn import metrics



def classifier(classifier_name, X_train, Y_train):
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TRAINING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
	if classifier_name == 'KNN':
		# Train KNeighborsClassifier Model
		KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
		KNN_Classifier.fit(X_train, Y_train)
		model = KNN_Classifier
	elif classifier_name == 'LGR':
		# Train LogisticRegression Model
		LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
		LGR_Classifier.fit(X_train, Y_train)
		model = LGR_Classifier
	elif classifier_name == 'BNB':
		# Train Gaussian Naive Baye Model
		BNB_Classifier = BernoulliNB()
		BNB_Classifier.fit(X_train, Y_train)
		model = BNB_Classifier
	elif classifier_name == 'DTC':            
		# Train Decision Tree Model
		DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
		DTC_Classifier.fit(X_train, Y_train)
		model = DTC_Classifier
	elif classifier_name == 'SVM':
		SVC_Classifier = SVC(probability=True,  kernel="rbf", C=2.8, gamma=4)
		SVC_Classifier.fit(X_train, Y_train)
		model = SVC_Classifier
	elif classifier_name == 'MLP':
		MLP_Classifier = MLP(hidden_layer_sizes = (30,))
		MLP_Classifier.fit(X_train, Y_train)
		model = MLP_Classifier
	else:
		print('ERROR: Unrecognized type of classifier')

	#evaluate(classifier_name, model, X_train, Y_train)
	return model


def evaluate(classifier_name, model, X, Y):
	
    scores = cross_val_score(model, X, Y, cv=10)
    Y_pre = model.predict(X)
    evaluate_sub(classifier_name, Y, Y_pre)
    print ("Cross Validation Mean Score:" "\n", scores.mean())
    

def evaluate_sub(classifier_name, Y, Y_pre):
	accuracy = metrics.accuracy_score(Y, Y_pre)
	confusion_matrix = metrics.confusion_matrix(Y, Y_pre)
	classification = metrics.classification_report(Y, Y_pre)
	print()
	print('============================== {} Model Evaluation =============================='.format(classifier_name))
	print()
	print ("Model Accuracy:" "\n", accuracy)
	print()
	print("Confusion matrix:" "\n", confusion_matrix)
	print()
	print("Classification report:" "\n", classification) 
	print()

	



