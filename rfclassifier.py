import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


class randomforestclassifier(object):
    def __init__(self, data):
        self.data = data

    def model(self):
        columns = ['connectedComponents', 'triangles', 'coefficient', 'egonetSize', 'corenumber']
        classfier_column = ['riskfactor']
        # print self.data
        X = self.data.as_matrix(columns)
        Y = self.data.as_matrix(classfier_column)
        self.randomforest(X,Y)

    def randomforest(self, X, Y):
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
        model = RandomForestClassifier(n_estimators=1000)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y)
        model.fit(x_train, y_train.ravel())
        y_true, y_pred = y_test, model.predict(x_test)
        print "-----Random Forest without GridSearch-----"
        print classification_report(y_true, y_pred)
        ##################################################

        ########## With gridsearch #######################
        grid_values = {
            'n_estimators': [200, 700],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y)
        clf = GridSearchCV(RandomForestClassifier(n_estimators=1000), param_grid=grid_values, scoring="f1", cv=5)
        clf.fit(x_train, y_train.ravel())
        y_true , y_pred = y_test, clf.predict(x_test)
        print "-----Random Forest with GridSearch-----"
        print classification_report(y_true, y_pred)
        ##################################################


