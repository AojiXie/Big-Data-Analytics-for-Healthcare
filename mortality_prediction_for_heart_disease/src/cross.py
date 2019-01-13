import models_partc
from sklearn.model_selection import KFold, ShuffleSplit

from numpy import mean
import numpy as np
from sklearn.metrics import *
import my_model

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS VALIDATION TESTS, OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
	kf = KFold(n_splits = k, random_state=RANDOM_STATE)
	kf.get_n_splits(X)

	accuracies = []
	aucs = []

	for train_index, test_index in kf.split(X):
		Y_pred = models_partc.logistic_regression_pred(X[train_index], Y[train_index], X[test_index])
		#Y_pred = my_model.my_classifier_predictions(X[train_index], Y[train_index], X[test_index])
		accuracy = accuracy_score(Y_pred, Y[test_index])
		auc_score = roc_auc_score(Y_pred, Y[test_index])
		accuracies.append(accuracy)
		aucs.append(auc_score)

	return np.mean(accuracies), np.mean(aucs)


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
	sskf = ShuffleSplit(n_splits=iterNo, test_size=test_percent, random_state=RANDOM_STATE)
	sskf.get_n_splits(X)
	accuracies = []
	aucs = []

	for train_index, test_index in sskf.split(X):
		Y_pred = models_partc.logistic_regression_pred(X[train_index], Y[train_index], X[test_index])
		#Y_pred = my_model.my_classifier_predictions(X[train_index],Y[train_index],X[test_index])
		accuracy = accuracy_score(Y_pred, Y[test_index])
		auc_score = roc_auc_score(sorted(Y_pred), sorted(Y[test_index]))
		accuracies.append(accuracy)
		aucs.append(auc_score)

	return np.mean(accuracies), np.mean(aucs)

def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

