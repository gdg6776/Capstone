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

warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
warnings.filterwarnings('ignore')

class ProcessingData(object):
    def __init__(self, process_data):
        self.process_data = process_data

    def datapreprocessing(self):
        # self.process_data = self.process_data[['connectedComponents', 'triangles', 'coefficient', 'egonetSize',
        #                                        'corenumber', 'riskfactor']]
        #
        # self.columns = [self.process_data.connectedComponents, self.process_data.triangles, self.process_data.coefficient,
        #                 self.process_data.egonetSize, self.process_data.corenumber]
        #
        # labelEncoder = preprocessing.LabelEncoder()
        # self.process_data.connectedComponents = labelEncoder.fit_transform(self.columns[0])
        # self.process_data.triangles = labelEncoder.fit_transform(self.columns[1])
        # self.process_data.coefficient = labelEncoder.fit_transform(self.columns[2])
        # self.process_data.egonetSize = labelEncoder.fit_transform(self.columns[3])
        # self.process_data.corenumber = labelEncoder.fit_transform(self.columns[4])
        return self.process_data



class classifydata(object):
    def __init__(self, data):
        self.data = data


    def classifier(self):

        columns = ['connectedComponents', 'triangles', 'coefficient', 'egonetSize','corenumber']
        classfier_column = ['riskfactor']
        # print self.data
        X = self.data.as_matrix(columns)
        Y = self.data.as_matrix(classfier_column)
        # print X
        # print Y

        self.Logistic_regression(X, Y)



    def Logistic_regression(self, X, Y):
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
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        model.fit(x_train, y_train.ravel())
        y_true, y_pred = y_test, model.predict(x_test)
        print "-----Logistic Regression without GridSearch-----"
        print classification_report(y_true, y_pred)
        ##################################################

        ######### With GridSearch #####################
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y)
        parameters = [{'penalty': ['l1'],
                       'C':[0.01, 0.1, 1, 5]},
                      {'penalty':['l2'], 'C': [0.01, 0.1, 1, 5] }]
        # param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        clf = GridSearchCV(linear_model.LogisticRegression(), parameters, cv=5, scoring="f1")
        clf.fit(x_train, y_train.ravel())
        y_true, y_pred = y_test, clf.predict(x_test)
        print "-----Logistic Regression with GridSearch-----"
        print classification_report(y_true, y_pred)
        ##################################################