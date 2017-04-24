#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ### 0.821387940842 0.892491467577
# create a classifier
clf = SVC(kernel="rbf", C = 10000)

t0 = time()
# train the classifier
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100]
clf.fit(features_train, labels_train)

# time taken for training
print "training time:", round(time()-t0, 3), "s"

# predict
pred = clf.predict(features_test)


chris_count = 0
for i in xrange(1, len(pred)):
	if pred[i] == 1:
		chris_count += 1

# find the accuracy
print chris_count
#accuracy = accuracy_score(pred[10], labels_test)

print "***********"
#print accuracy
#########################################################