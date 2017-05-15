#!/usr/bin/python

import sys
import pickle
import math
sys.path.append("../tools/")
import pandas as pd
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

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
df.fillna(0, inplace=True)
#print df
print "Total POI's: {}".format(len(df[df['poi'] == 1]))
print "Total Non POI's: {}".format(len(df[df['poi'] == 0]))

# Here, we can say that, we have non poi's almost 7 times the number of poi's which implicitly means
# that, we have less training data to learn from i.e. less number of dependent variables to learn from

# 
#print df.describe().transpose()
#print df.head(5)
#print df.columns[df.columns == 'poi']
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
create_plots_for_outliers('total_payments', 'total_stock_value')
create_plots_for_outliers('from_poi_to_this_person', 'from_this_person_to_poi')
create_plots_for_outliers('bonus', 'salary')

# We can clearly see that 'TOTAL' is a spreadsheet quirk and should be removed
data_dict.pop('TOTAL', 0)

# Let's also identify the other outliers by looking at the plots, we will clean them later


#removing outliers
def remove_outlier(keys):
	""" removes list of outliers keys from dict object """
	non_outlier_data_dict = data_dict
	for key in keys:
		non_outlier_data_dict.pop(key, 0)

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
remove_outlier(outliers)

exit(0)
### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

'''
# scatter plot
for point in data:
	salary = point[0]
	bonus = point[1]
	plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
'''

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit()

exit(0)

### Task 4: Try a variety of classifiers
'''

'''

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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