import utils
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.svm import LinearSVC
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features(filtered_events, feature_map):
	#TODO: complete this
	events_to_idx = pd.merge(filtered_events, feature_map, on='event_id')
	events_to_idx = events_to_idx[['patient_id', 'idx', 'value']]
	events_to_idx = events_to_idx.dropna()

	events_sum = events_to_idx[events_to_idx['idx'] < 2680]
	events_count = events_to_idx[events_to_idx['idx'] >= 2680]

	events_counts = events_count.groupby(['patient_id', 'idx']).agg('count')
	# events_counts.columns = ['patient_id', 'event_id', 'value']

	events_sums = events_sum.groupby(['patient_id', 'idx']).agg('sum')
	# events_sums.columns = ['patient_id', 'event_id', 'value']

	total_events = pd.concat([events_counts, events_sums])
	total_events.columns = ['value']
	total_events = total_events.reset_index()

	##min- max
	total_events1 = total_events[['idx', 'value']]

	# min_events_value = total_events1.groupby(['idx']).min()

	max_events_value = total_events1.groupby(['idx']).max()

	# maxsubmin = max_events_value.sub(min_events_value)
	max_events_value = max_events_value.reset_index()
	max_events_value.columns = ['idx', 'max_value']
	# min_events_value = min_events_value.reset_index()
	# min_events_value.columns = ['idx', 'min_value']
	# maxsubmin = maxsubmin.reset_index()
	# maxsubmin.columns = ['idx', 'max-min']

	# normalized_df = pd.merge(min_events_value, maxsubmin, on='idx')
	# normalized_df = pd.merge(normalized_df, max_events_value, on='idx')

	df1 = pd.merge(total_events, max_events_value, on='idx')

	df1_not_zero = df1[df1['max_value'] != 0]
	df1_not_zero['value'] = df1_not_zero['value'] / df1_not_zero['max_value']

	df1_zero = df1[df1['max_value'] == 0]

	# df1_zero_events = df1_zero['idx'].value_counts()
	# df1_zero_events = df1_zero_events.reset_index()
	# df1_zero_events.columns = ['idx', 'counts']
	# df1_zero = pd.merge(df1_zero, df1_zero_events, on='idx')
	df1_zero['value'] = 1.0
	# df1_zero = df1_zero[['patient_id', 'idx', 'value', 'min_value', 'max-min']]

	aggregated_events = pd.concat([df1_zero, df1_not_zero])

	ggregated_events = pd.concat([df1_zero, df1_not_zero])

	aggregated_events = aggregated_events[['patient_id', 'idx', 'value']]
	aggregated_events.columns = ['patient_id', 'feature_id', 'feature_value']


# create features

	aggregated_events['merged'] = aggregated_events.apply(lambda row: (row['feature_id'], row['feature_value']), axis=1)

	# aggregated_events['merged'] = aggregated_events.set_index('feature_id')['feature_value'].T.apply(tuple)
	# aggregated_events['merged'] = aggregated_events['feature_id'].astype(str) +':'+aggregated_events['feature_value'].astype(str)
	# aggregated_events['merged'] = aggregated_events['merged'].astype(float)
	patient_features = aggregated_events.groupby('patient_id')['merged'].apply(lambda x: x.tolist()).to_dict()

	deliverable1 = open('../deliverables/test_features.txt', 'wb')

	#lines = 0

	for key in sorted(patient_features):
		line = "%d" %(key)

		for value in sorted(patient_features[key]):
			merged = "%d:%.6f" %(value[0], value[1])
			line = line +" " + merged
		#lines+=1

		deliverable1.write((line + " " + "\n").encode())

		#if lines >=633:
			#break


	'''
	You can use any model you wish.
	
	input: X_train, Y_train, X_test
	output: Y_pred
	'''




def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
	#regr1 = DecisionTreeClassifier(max_depth= 5)
	regr1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=1000)
	#regr1 = DecisionTreeClassifier(max_depth = 5)
	#regr2 = AdaBoostClassifier(RandomForestClassifier(n_estimators= 100,random_state=123456,max_depth = 5), n_estimators= 500)
	#regr1 = LinearSVC(C = 5)
	regr2 = BaggingClassifier(regr1, n_estimators =10)
	regr2.fit(X_train, Y_train)
	Y_pred = regr2.predict(X_test)


	return Y_pred


def main():
	events =  pd.read_csv('../data/test/events.csv')
	feature_map = pd.read_csv('../data/test/event_feature_map.csv')
	my_features(events,feature_map)
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	X_test, Y_test = utils.get_data_from_svmlight("../deliverables/test_features.txt")
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	#Y_pred =np.round(Y_pred)
	print(Y_pred)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	