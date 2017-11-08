"""
References -  http://librimind.com/2015/07/logistic-regression-with-python/
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
warnings.filterwarnings('ignore')


class classifydata(object):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data


    def classifier(self):

        columns = ['connectedComponents', 'triangles', 'coefficient', 'egonetSize','corenumber']
        classfier_column = ['riskfactor']

        X_train = self.train_data.as_matrix(columns)
        Y_train = self.train_data.as_matrix(classfier_column)

        X_test = self.test_data.as_matrix(columns)
        Y_test = self.test_data.as_matrix(classfier_column)

        self.Logistic_regression(X_train, Y_train, X_test, Y_test)



    def Logistic_regression(self, X_train, Y_train, X_test, Y_test):
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
        model = linear_model.LogisticRegression(C=10000)
        model.fit(X_train, Y_train)
        y_true, y_pred = Y_test, model.predict(X_test)
        print "-----Logistic Regression without GridSearch-----"
        print classification_report(y_true, y_pred)
        ##################################################

        ######### With GridSearch #####################
        # x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y)
        parameters = [{'penalty': ['l1'],
                       'C':[0.01, 0.1, 1, 5]},
                      {'penalty':['l2'], 'C': [0.01, 0.1, 1, 5] }]
        clf = GridSearchCV(linear_model.LogisticRegression(), parameters, cv=10, scoring="f1")
        clf.fit(X_train, Y_train.ravel())
        y_true, y_pred = Y_test, clf.predict(X_test)
        print "-----Logistic Regression with GridSearch-----"
        print classification_report(y_true, y_pred)
        ##################################################