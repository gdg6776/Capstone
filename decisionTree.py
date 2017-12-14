import numpy as np
from sklearn import cross_validation
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE


class decisiontreeclassifier(object):
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
        self.decisiontree(X_train, Y_train, X_test, Y_test)

    def decisiontree(self,X_train, Y_train, X_test, Y_test):
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
        model = tree.DecisionTreeClassifier()

        ######### Without GridSearch #####################
        model.fit(X_train, Y_train.ravel())
        y_true, y_pred = Y_test, model.predict(X_test)
        print "-----Decision Tree without GridSearch-----"
        print classification_report(y_true, y_pred)
        ##################################################


        ######### With GridSearch #####################
        grid_values = {

            #'max_depth': np.arange(3, 10), 
            'criterion' : ['gini', 'entropy'],
            'class_weight':['balanced']
        }
        clf = GridSearchCV(model, param_grid=grid_values, scoring="f1", cv=5)
        clf.fit(X_train, Y_train.ravel())
        y_true, y_pred = Y_test, clf.predict(X_test)
        print "-----Decision Tree with GridSearch----"
        print clf.best_params_
        #print clf.coef_

        print classification_report(y_true, y_pred)
        ##################################################

        ######### RFE ####################################
        # rfe = RFE(model, 5)
        # rfe = rfe.fit(X_train, Y_train.ravel())
        # y_true, y_pred = Y_test, rfe.predict(X_test)
        # print "-----RFE-----"
        # print classification_report(y_true, y_pred)
        ##################################################



