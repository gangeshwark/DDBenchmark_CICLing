import tensorflow as tf
import numpy as np

import load_data

np.random.seed(100000)

X_train_text, X_test_text, X_train_audio, X_test_audio, X_train_gest, X_test_gest, X_train_video, X_test_video, Y_train, Y_test = load_data.load()

print(X_train_audio.shape, X_test_audio.shape)

if __name__ == '__main__':
    from sklearn import svm, tree, ensemble

    # f = np.concatenate((X_train_audio, X_train_text, X_train_video, X_train_gest), axis=1)
    X_train = np.concatenate((X_train_audio, X_train_text, X_train_video, X_train_gest), axis=1)
    X_test = np.concatenate((X_test_audio, X_test_text, X_test_video, X_test_gest), axis=1)
    print(X_train.shape)

    print("SVM")
    clf = svm.SVC()
    clf.fit(X_train, np.argmax(Y_train, axis=1))
    # print(clf.support_vectors_)

    a = clf.predict(X_test)
    print(a.shape)
    y_true = np.argmax(Y_test, 1)
    print(np.mean(y_true == a))

    print("DecisionTreeClassifier")
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, np.argmax(Y_train, axis=1))
    a = clf.predict(X_test)
    print(a.shape)
    y_true = np.argmax(Y_test, 1)
    print(np.mean(y_true == a))

    print("RandomForestClassifier")
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train, np.argmax(Y_train, axis=1))
    a = clf.predict(X_test)
    print(a.shape)
    y_true = np.argmax(Y_test, 1)
    print(np.mean(y_true == a))
