#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments','exercised_stock_options','shared_receipt_with_poi','total_stock_value','expenses','from_this_person_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

for item in data_dict:
	print data_dict[item]
	break

def resample(data, uniformity=1.0, sample_size=1, random_state=0):
	import random
	import numpy as np
	
	orig_size = len(data)
	samp_size = int(np.floor(sample_size * orig_size))

	class_dict = {}

	for _, value in data.iteritems():
		poi = value['poi']
		if poi not in class_dict:
			class_dict[poi] = []
		class_dict[poi].append(value)

	n_classes = len(class_dict.keys())

	result = {}

	X_prime = []
	y_prime = []

	for i in range(samp_size):
		if uniformity >= random.random():
			idx = random.randint(0, n_classes-1)
			rand_class = class_dict.keys()[idx]
			rand_offs = random.randint(0, len(class_dict[rand_class]) - 1)
			result[i] = class_dict[rand_class][rand_offs]
		else: 
			idx = random.randint(0, orig_size-1)
			result[i] = data[data.keys()[idx]]

	return result

data_dict = resample(data_dict, 0.6, 1.5, 42)

my_dataset = data_dict

for item, value in my_dataset.iteritems():
	features_list = value.keys()
	break

tmp_list = ['poi']
for feature in features_list:
	if feature not in ['email_address', 'poi']:
		tmp_list.append(feature)

features_list = tmp_list

# import csv
# with open('dict.csv', 'wb') as f:
# 	writer = csv.DictWriter(f, fieldnames=column_csv)
# 	writer.writeheader()
# 	for _, data in my_dataset.iteritems():
# 		writer.writerow(data)



	

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

def resample(X, y, uniformity, random_state, sample_size):
	import random
	import numpy as np
	
	orig_size = len(X)
	samp_size = int(np.floor(sample_size * orig_size))

	class_dict = {}

	for index, label in enumerate(y):
		if label not in class_dict:
			class_dict[label] = []
		class_dict[label].append(X[index])

	n_classes = len(class_dict.keys())

	X_prime = []
	y_prime = []

	for i in range(samp_size):
		if uniformity >= random.random():
			idx = random.randint(0, n_classes-1)
			rand_class = class_dict.keys()[idx]
			rand_offs = random.randint(0, len(class_dict[rand_class]) - 1)
			X_prime.append(class_dict[rand_class][rand_offs])
			y_prime.append(rand_class)
		else: 
			idx = random.randint(0, orig_size-1)
			X_prime.append(X[idx])
			y_prime.append(y[idx])

	return np.asarray(X_prime), np.asarray(y_prime)

features, labels = resample(features, labels, 0.6, 42, 1.5)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)