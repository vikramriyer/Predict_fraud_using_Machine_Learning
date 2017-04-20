from sklearn import svm
X = [[0,0],[1,1]]
Y = [0,1]
clf = svm.SVC()
print clf.fit(X, Y)

print clf.predict([2.,2.])
print clf.support_vectors_
print clf.support_
print clf.n_support_
