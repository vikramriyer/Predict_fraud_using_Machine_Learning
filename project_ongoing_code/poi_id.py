#!/usr/bin/python

import sys
import pickle
import math
import pandas as pd
import numpy as np
import operator

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import matplotlib.pyplot as plt
from numpy import mean

from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import KFold, cross_val_score
from sklearn import preprocessing
from sklearn.decomposition import PCA, NMF

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import chi2

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

### Task 1: Select what features you'll use.
'''
Currently using all the features, should remove ones which are not necessary
'''

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus','long_term_incentive',
				'deferral_payments', 'expenses','deferred_income',
				'director_fees','loan_advances','other', 'restricted_stock', 
				'exercised_stock_options','restricted_stock_deferred', 
				'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# preparing data for analysis
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=np.float)
df.reset_index(level=0, inplace=True)
df.rename(columns={'index':'name'}, inplace=True)
#print df.columns
#print df.describe()
#print df.head()

# exploring data
group_poi = df.groupby('poi') #This will group the df by poi
'''
print "number of features: ", len(df.keys()), "\n"
print group_poi['poi'].agg([len]) #This will help us find how many are there who are POI i.e. 1 and how many are not i.e. 0

print group_poi['bonus'].agg([len])
print group_poi['bonus',
    'salary',
    'exercised_stock_options',
    'shared_receipt_with_poi',
    'loan_advances',
    'to_messages'].agg([np.max])
print df.describe().transpose()
'''
#df.fillna(0, inplace=True)
#print df
print "Total POI's: {}".format(len(df[df['poi'] == 1]))
print "Total Non POI's: {}".format(len(df[df['poi'] == 0]))

# Here, we can say that, we have non poi's almost 7 times the number of poi's which implicitly means
# that, we have less training data to learn from i.e. less number of dependent variables to learn from

print df.describe().transpose()
print df.head(5)
print df.columns[df.columns == 'poi']
# By looking at the count section, we can say that, some columns/features do not have useful info or
# have NaN, these include; loan_advances, restricted_stock_deferred, director_fees, deferral_payments, deferred_income

### Task 2: Remove outliers
# Let's start looking at some visualizations to find outliers in the dataset, which features have
# most number of outliers and how we can eliminate if we should.

#plotting outliers
import matplotlib.pyplot as plt
def create_plots_for_outliers(feature_1, feature_2):
	""" Plot with flag = True in Red """
	data = featureFormat(data_dict, [feature_1, feature_2, 'poi'])
	for point in data:
		x = point[0]
		y = point[1]
		poi = point[2]
		if poi:
			color = 'red'
		else:
			color = 'blue'
		plt.scatter(x, y, color=color)
	plt.xlabel(feature_1)
	plt.ylabel(feature_2)
	plt.show()

# By observing the above plots, it makes sense to check the following features for outliers
#create_plots_for_outliers('total_payments', 'total_stock_value')
#create_plots_for_outliers('from_poi_to_this_person', 'from_this_person_to_poi')
#create_plots_for_outliers('bonus', 'salary')

# Let's also identify the other outliers by looking at the plots, we will clean them later
#total_payments vs total_stock_value
#total_payments > 1
total_payments_list = df['total_payments'].tolist()
total_payments_list.sort(reverse=True)
max_val = max(total_payments_list)
print total_payments_list[0:5]
print max_val
print df[df.total_payments >= 103559793.0].name
# Here, we can say that Lay Kenneth is a valid data point. However, we can clearly 
# see that 'TOTAL' is a spreadsheet quirk and should be removed

#from_poi_to_this_person vs from_this_person_to_poi
#from_poi_to_this_person > 300 as well as from_this_person_to_poi > 200
print df[df.from_poi_to_this_person > 300].name
print df[df.from_this_person_to_poi > 200].name
#The above results suggest that there is nothing wrong with the data here. These sdata points will
#be analyzed further if anything suspicious is found

#bonus vs salary
#bonus > 0.2
print df[df.bonus == max(df['bonus'].tolist())].name
#The above result too results on 'TOTAL' which we will remove

#Apart from the above outliers, there are some more which need attention either because they lack
#sensible data that can be used for analysis or they are wrong
print df[df.name == 'THE TRAVEL AGENCY IN THE PARK']
print df[df.name == 'LOCKHART EUGENE E']
#The above data points are not much useful because most of the fields are 'NaN', so we will drop them

#removing outliers
def remove_outlier(keys):
	""" removes list of outliers keys from dict object """
	non_outlier_data_dict = data_dict
	for key in keys:
		non_outlier_data_dict.pop(key, 0)
	return non_outlier_data_dict

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
non_outlier_data_dict = remove_outlier(outliers)

'''
NAN programatically
# Since we saw a lot of features for records having 'NaN' which we converted to '0', 
# it is worth finding out the number of 'NaN' for each record
all_features = df.columns.tolist()
nan_features = {}
for feature in all_features:
	nan_features[feature] = 0
	for k,v in data_dict.iteritems():
		if feature == 'name':
			continue
		if v[feature] == 'NaN':
			nan_features[feature] += 1

print sorted(nan_features.items(), key=operator.itemgetter(1), reverse=True)
# With the above result we can see that, loan_advances, director_fees, restricted_stock_deferred,
deferral_payments, deferred_income, long_term_incentive are the features which have more than 50% of
values to be 'NaN'
'''

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

# Let's find the fraction of emails from a person and to a person, and add this as a feature
def find_fraction(user_messages, total_messages):
	if user_messages != 'NaN' and total_messages != 'NaN':
		fraction = user_messages/float(total_messages)
	else:
		fraction = 0
	return fraction

def add_all_money(*all_financial_features):
	total_money = 0
	for feature in all_financial_features:
		if feature != 'NaN':
			total_money += feature
	return total_money

financial_features = ['salary', 'total_stock_value', 
					  'exercised_stock_options', 'bonus']

for name, data in my_dataset.iteritems():
	data['fraction_of_messages_from_pois'] = find_fraction(data['from_poi_to_this_person'],
												data['to_messages'])
	data['fraction_of_messages_to_pois'] = find_fraction(data['from_this_person_to_poi'],
												data['from_messages'])
	data['total_assets'] = add_all_money(data['salary'],
										 data['total_stock_value'],
										 data['exercised_stock_options'],
										 data['bonus'])

# We can add one more measure of a person, i.e. the wealth which is the sum of all the financial\
# features present in the dataset. I personally found it very interesting to know how much money
# on the whole people made in the year 2001-02
name_assets_map = {}
for k, v in my_dataset.iteritems():
	name_assets_map[k] = [v['total_assets'], v['poi']]

top_10_guns = sorted(name_assets_map.items(), key=operator.itemgetter(1), reverse=True)[:10]
print "The top 10 money makers: {}".format(top_10_guns)

# let's add the newly created features to the features list
features_list.extend(('total_assets', 'fraction_of_messages_to_pois', 'fraction_of_messages_from_pois'))

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

# scatter plot
for point in data:
	from_poi = point[1]
	to_poi = point[2]
	plt.scatter( from_poi, to_poi )
	if point[0] == 1:
		plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
#plt.show()

labels, features = targetFeatureSplit(data)

# Now that we have added 3 more features to existing, let's find the 10 best features that can be used
from sklearn.feature_selection import SelectKBest
k_best = SelectKBest(k=10)
k_best.fit(features, labels)

results_list = zip(k_best.get_support(), features_list[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
k_best_features = [i[1] for i in results_list]
print k_best_features

# poi is always expected feature at the 0th position, hence let's add it
k_best_features.insert(0, 'poi')

data = featureFormat(my_dataset, k_best_features, sort_keys = True)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.1, random_state=42)

### Task 4: Try a variety of classifiers

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
print "CLASSIFIERS"
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
print accuracy_score(labels_test, pred)
print precision_score(labels_test, pred)
#OR
#print clf.score(features_test, labels_test)
print '-x-x-x-Naive Bayes-x-x-x-'

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=5)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print accuracy_score(labels_test, pred)
print precision_score(labels_test, pred)
print '-x-x-x-Decision Trees-x-x-x-'

from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', C = 10000)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print accuracy_score(labels_test, pred)
print precision_score(labels_test, pred)
print '-x-x-x-SVM-x-x-x-'

exit(0)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
