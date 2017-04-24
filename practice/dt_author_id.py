#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn import tree
from sklearn.metrics import accuracy_score

# create classifier
clf_40 = tree.DecisionTreeClassifier(min_samples_split=40)

# train the classifier
clf_40.fit(features_train, labels_train)

# predict test label based on test features
pred_40 = clf_40.predict(features_test)

# accuracy
acc_40 = accuracy_score(pred_40, labels_test)

print "acc 40: {}".format(acc_40)
print "total features: {}".format(len(features_train[0]))

#########################################################


