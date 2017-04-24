#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
def k_neigh():
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.metrics import accuracy_score
	t0 = time()
	#features_train = features_train[:len(features_train)/100] 
	#labels_train = labels_train[:len(labels_train)/100]

	print "initializing the classifier"
	clf = KNeighborsClassifier(n_neighbors = 5)

	print "training the classifier"
	clf.fit(features_train, labels_train)
	print "training time:", round(time()-t0, 3), "s"

	t1 = time()
	print "predicting test labels based on training set"
	pred = clf.predict(features_test)
	print "prediction time:", round(time()-t0, 3), "s"

	acc = accuracy_score(pred, labels_test)
	print "acc: {}".format(acc)

def ada_boost():
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.metrics import accuracy_score
	t0 = time()
	#features_train = features_train[:len(features_train)/100] 
	#labels_train = labels_train[:len(labels_train)/100]

	print "initializing the classifier"
	clf = AdaBoostClassifier(n_estimators=10)

	print "training the classifier"
	clf.fit(features_train, labels_train)
	print "training time:", round(time()-t0, 3), "s"

	t1 = time()
	print "predicting test labels based on training set"
	pred = clf.predict(features_test)
	print "prediction time:", round(time()-t0, 3), "s"

	acc = accuracy_score(pred, labels_test)
	print "acc: {}".format(acc)

def random_forest():
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score
	t0 = time()
	#features_train = features_train[:len(features_train)/100] 
	#labels_train = labels_train[:len(labels_train)/100]

	print "initializing the classifier"
	clf = RandomForestClassifier(n_estimators=7)

	print "training the classifier"
	clf.fit(features_train, labels_train)
	print "training time:", round(time()-t0, 3), "s"

	t1 = time()
	print "predicting test labels based on training set"
	pred = clf.predict(features_test)
	print "prediction time:", round(time()-t0, 3), "s"

	acc = accuracy_score(pred, labels_test)
	print "acc: {}".format(acc)


k_neigh()
print "############################################"
ada_boost()
print "############################################"
random_forest()

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    print "error occured"
#plt.show()