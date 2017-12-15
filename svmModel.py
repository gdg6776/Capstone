import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, cross_validation
from sklearn import metrics

from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import SVR

class svmMod(object):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def model(self):
        columns = ['connectedComponents', 'triangles', 'coefficient', 'egonetSize', 'corenumber']
        classfier_column = ['riskfactor']

        X_train = self.train_data.as_matrix(columns)
        Y_train = self.train_data.as_matrix(classfier_column)

        X_test = self.test_data.as_matrix(columns)
        Y_test = self.test_data.as_matrix(classfier_column)

        self.supportVectorMachines(X_train, Y_train, X_test, Y_test)

    def supportVectorMachines(self, X_train, Y_train, X_test, Y_test):
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
        model = SVC()

        ######### Without GridSearch #####################
        model.fit(X_train, Y_train)
        y_true, y_pred = Y_test, model.predict(X_test)
        print "-----Support Vector Machine without GridSearch-----"
        print classification_report(y_true, y_pred)
        ##################################################


        ########## With gridsearch #######################
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10], 'class_weight':['balanced']}

        # parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        #  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        
        clf = GridSearchCV(model, parameters, scoring="f1",cv=5)
        clf.fit(X_train, Y_train.ravel())
        y_true, y_pred = Y_test, clf.predict(X_test)
        print "-----Support Vector Machine with GridSearch-----"
        print classification_report(y_true, y_pred)
        ####################################################


        ########## RFE #######################
        params = clf.best_params_
        # {'kernel': 'linear', 'C': 10, 'class_weight': 'balanced'}
        estimator = SVC(kernel=params['kernel'], C=params['C'], class_weight=params['class_weight'])
        rfe = RFE(estimator, n_features_to_select=1, step=1)
        rfe = rfe.fit(X_train, Y_train.ravel())
        y_true, y_pred = Y_test, rfe.predict(X_test)
        features = ['connectedComponents', 'triangles', 'coefficient', 'egonetSize', 'corenumber']
        sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), features))
        feature_selected = dict(zip(rfe.ranking_, features))
        result = [feature_selected[key] for key in sorted(feature_selected.keys())]
        ####################################################


        for numbers in range(len(result), 0, -1):
            X_train = self.train_data.as_matrix(result[:numbers])
            X_test = self.test_data.as_matrix(result[:numbers])
            estimator.fit(X_train, Y_train)
            y_true, y_pred = Y_test, estimator.predict(X_test)
            print "-----SVM-----"
            print "features - " + str(result[:numbers])
            print classification_report(y_true, y_pred)