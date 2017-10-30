import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, cross_validation
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class svmMod(object):
    def __init__(self, data):
        self.data = data

    def model(self):
        columns = ['connectedComponents', 'triangles', 'coefficient', 'egonetSize', 'corenumber']
        classfier_column = ['riskfactor']
        # print self.data
        X = self.data.as_matrix(columns)
        Y = self.data.as_matrix(classfier_column)

        self.supposrtVectorMachines(X,Y)

    def supposrtVectorMachines(self, X, Y):
        """
               This is the function which calculates the Logistic regression and draws graphs related to it.
               :param X: This is the matrix of all the columns except the classifier column of
                                                          training data.
               :param Y: This is the matrix of classifier columns of training data.
               :param X_test:This is the matrix of all the columns except the classifier column of
                                                      testing data.
               :param Y_test: This is the matrix of classifier columns of testing data.
               :return:None
               """
        np.set_printoptions(suppress=True)

        ######### Without GridSearch #####################
        model = svm.SVC()
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        model.fit(x_train, y_train.ravel())
        y_true, y_pred = y_test, model.predict(x_test)
        print "-----Support Vector Machine without GridSearch-----"
        print classification_report(y_true, y_pred)
        ##################################################


        ########## With gridsearch #######################
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svc = svm.SVC()
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y)
        clf = GridSearchCV(svc, parameters, scoring="f1",cv=5)
        clf.fit(x_train, y_train.ravel())
        y_true, y_pred = y_test, clf.predict(x_test)
        print "-----Support Vector Machine with GridSearch-----"
        print classification_report(y_true, y_pred)
        ########## #########################################