import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.decomposition import PCA

iris = load_iris()
print iris.data.shape
print iris.target.shape

features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size = 0.3, random_state = 0)
'''
pca = PCA(n_components=2)
pca.fit(features_train)
pca.transform(features_train)
'''

clf = svm.SVC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

print accuracy_score(pred, labels_test)

pca = PCA(n_components=2)
pca.fit(features_train)
pca.transform(features_train)
pca.transform(features_test)

clf1 = svm.SVC()
clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)

print accuracy_score(pred1, labels_test)