def NBAccuracy(classifier, pred, labels_test):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test) #TODO
    return accuracy
