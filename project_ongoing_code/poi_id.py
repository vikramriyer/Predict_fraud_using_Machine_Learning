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

############################ <Task 1: Select what features you'll use.> ############################
'''
- Since we are not aware as of now about the performace of each feature in
predicting whether or not a person is poi, we will take all the features and
later skim down the ones which are less relevant/informative.
- 'features_list' is a list of strings, each of which is a feature name.
'''
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments',
       'total_payments', 'exercised_stock_options', 'bonus',
       'restricted_stock', 'shared_receipt_with_poi',
       'restricted_stock_deferred', 'total_stock_value', 'expenses',
       'loan_advances', 'from_messages', 'other',
       'from_this_person_to_poi', 'director_fees',
       'deferred_income', 'long_term_incentive',
       'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# load the dictionary into a dataframe
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=np.float)

# exploring data: our data is now ready to be explored
print "Total number of Data points: '{}' and Features: '{}'".format(df.shape[0], df.shape[1])
print df.describe().transpose()
print df.head()
print df.shape
print "Total POI's: {}".format(len(df[df['poi'] == 1]))
print "Total Non POI's: {}".format(len(df[df['poi'] == 0]))

# change the index from name of the person
# to int indexes, we may need the name as a separate field
df.reset_index(level=0, inplace=True)
df.rename(columns={'index':'name'}, inplace=True)

'''
Inferences after exploring data
 - less number of data points to train -> 146
 - the count in the describe() output shows that there are lot of NaN values
 - lot of non-poi's than poi's
'''

# print the features in the descending order of number of NaN's
nan_features = {}
for feature in features_list:
    nan_features[feature] = 0
    for k,v in data_dict.iteritems():
        if feature == 'name':
            continue
        if v[feature] == 'NaN':
            nan_features[feature] += 1

print sorted(nan_features.items(), key=operator.itemgetter(1), reverse=True)
'''
We can remove loan_advances, director_fees, restricted_stock_deferred, 
deferral_payments, deferred_income, long_term_incentive as more than
50% points are NaN
'''
remove_features = ['loan_advances','director_fees','restricted_stock_deferred',
        'deferral_payments','deferred_income', 'long_term_incentive']
for feature in remove_features:
    features_list.remove(feature)

############################ <Task 2: Remove outliers> ############################
# Let's start looking at some visualizations to find outliers in the dataset, which features have
# most number of outliers and how we can eliminate if we should.
# Note: We are mostly interested in numerical features

# plotting outliers
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

# plot the graph
#create_plots_for_outliers('total_payments', 'total_stock_value')
#create_plots_for_outliers('from_poi_to_this_person', 'from_this_person_to_poi')
#create_plots_for_outliers('bonus', 'salary')

'''
By observing the above plots, I am eager on finding out the person whose
total_payments is far ahead of others!
'''
print df.sort('total_payments', ascending=False).head(10)

'''
Unfortunately, it is not a person but a spreadsheet calculation named 'TOTAL' 
that calculates the total sum of all the payments
'''

print df[df.from_poi_to_this_person > 300].name
print df[df.from_this_person_to_poi > 200].name
'''
Nothing seems to be wrong with the data here. We can analyze them later if needed.
'''

print df[df.bonus == max(df['bonus'].tolist())].name
'''
Same as before, 'TOTAL' will be removed
'''

'''
Apart from the above outliers, there are some more which need attention either because they lack
sensible data that can be used for analysis or they are wrong
'''

print df[df.name == 'THE TRAVEL AGENCY IN THE PARK']
print df[df.name == 'LOCKHART EUGENE E']
'''
The above data points are not much useful because most of the fields are 'NaN', so we will drop them
'''
#removing outliers
def remove_outlier(keys):
    """ removes list of outliers keys from dict object """
    for key in keys:
        data_dict.pop(key, 0)

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
remove_outlier(outliers)

############################ <Task 3: Create new feature(s)> ############################
'''
We will engineer 3 features which are intuitive
- As per the mini projects, we can better find out the fraction of messages from and to
rather than considering the sum of the total messages. So new features will be,
1. fraction_of_messages_from_pois
2. fraction_of_messages_from_pois
- Another feature that may give us some insights about who made the most money
and determine whether the person is a 'poi'
3. total_assets
'''


### Store to my_dataset for easy export below.
my_dataset = data_dict

# returns fraction of emails from a person and to a person, and add this as a feature
def find_fraction(user_messages, total_messages):
    if user_messages != 'NaN' and total_messages != 'NaN':
        fraction = user_messages/float(total_messages)
    else:
        fraction = 0
    return fraction

# add all the financial features and returns total asset
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

'''
Now that we have engineered the new features, let's find out who were the top money makers
'''
name_assets_map = {}
for k, v in my_dataset.iteritems():
    name_assets_map[k] = [v['total_assets'], v['poi']]

top_10_guns = sorted(name_assets_map.items(), key=operator.itemgetter(1), reverse=True)[:10]
print "The top 10 money makers: {}".format(top_10_guns)
print

# let's add the newly created features to the features list
features_list.extend(('total_assets', 'fraction_of_messages_to_pois', 'fraction_of_messages_from_pois'))

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

# scatter plot to see how the newly added fractions of msgs from and to pois
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
plt.ylabel("fraction of emails this person sends to poi")
#plt.show()

labels, features = targetFeatureSplit(data)

# Let's select the k-best features, where k = 10
def select_k_best_features():
    '''
    Returns a list of the top 10 features
    k is defaulted to 10
    '''
    for k_val in range(1, len(features_list)):
        k_best = SelectKBest(k=k_val)
        k_best.fit(features, labels)
        scores = k_best.scores_
        unsorted_pairs = zip(features_list[1:], scores)
        sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
        k_best_features = dict(sorted_pairs[:k_val])
        final_features = k_best_features.keys()
    
        # poi is always expected feature at the 0th position, hence let's add it
        final_features.insert(0, 'poi')
        print "------->>> {}".format(final_features)

        feature_dict[k_val] = final_features

feature_dict = {}
select_k_best_features()

def print_and_return_scores(classifier, tuned=False):
    '''
    Prints the accuracy, precision and recall scores so as to make optimal choices
    '''
    if tuned:
        accuracy = cross_val_score(classifier, features, labels, cv=10, scoring='accuracy').mean()
        precision = cross_val_score(classifier, features, labels, cv=10, scoring='precision').mean()
        recall = cross_val_score(classifier, features, labels, cv=10, scoring='recall').mean()
        print "Accuracy: {}".format(accuracy)
        print "Precision: {}".format(precision)
        print "Recall: {}".format(recall)
        return accuracy, precision, recall
    else:
        classifier.fit(features_train, labels_train)
        pred = classifier.predict(features_test)
        print "Accuracy: {}".format(accuracy_score(labels_test, pred))
        print "Precision: {}".format(precision_score(labels_test, pred))
        print "Recall: {}".format(recall_score(labels_test, pred))

max_scores = {'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
def get_max(score_tuple):
    if score_tuple[1] > max_scores['precision'] and score_tuple[2] > max_scores['recall'] \
        and score_tuple[0] > max_scores['accuracy']:
        max_scores['precision'] = score_tuple[1]
        max_scores['recall'] = score_tuple[2]
        max_scores['accuracy'] = score_tuple[0]
        return True

total_k_features = 0
for i in range(1,len(feature_dict)):
    print "***************************************Features: {}**********************************************".format(i)
    data = featureFormat(my_dataset, feature_dict[i], sort_keys = True)
    #data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    ############################ <Task 4: Try a variety of classifiers> ############################
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.1, random_state=42)

    print "Before Tuning..."
    print 
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier()
    svm = SVC()
    print "KNN"
    print_and_return_scores(knn)
    print
    print "Decision Trees"
    print_and_return_scores(dt)
    print
    print "Support Vector Machines"
    print_and_return_scores(svm)
    print 

    ############################ <Task 5: Tune your classifier to achieve better than .3 precision and recall> ############################
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    # Example starting point. Try investigating other evaluation techniques!

    '''
    Since this is a small dataset, it will far better to use K-fold cross-validation
    rather than train-test-split as we will have (k-1) types of train test split work
    for us instead of a single one.
    Though this is compute heavy, with this small a dataset, it should not take a lot
    of time
    Setting the second parameter to True will use kfold cross validation
    '''

    print "After Tuning..."
    svm = SVC(kernel='rbf', gamma='auto', C=1.0)
    knn = KNeighborsClassifier(n_neighbors=7, n_jobs=3)
    dt = DecisionTreeClassifier(max_depth=11, random_state=51)
    print

    print "Support Vector Machines"
    if get_max(print_and_return_scores(svm, True)):
        clf = svm
        total_k_features = i
    print 

    print "KNN"
    if get_max(print_and_return_scores(knn, True)):
        clf = knn
        total_k_features = i
    print 

    print "Decision Trees"
    if get_max(print_and_return_scores(dt, True)):
        clf = dt
        total_k_features = i
    print "*******************************************************************************************************"

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

print clf
print
print max_scores
print "k: {}".format(total_k_features)
print "The features: {}".format(feature_dict[total_k_features])
print "The features: {}".format(feature_dict[total_k_features+1])
dump_classifier_and_data(clf, my_dataset, features_list)
