import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


class randomforestclassifier(object):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def model(self):
        columns = ['connectedComponents', 'triangles', 'coefficient', 'egonetSize', 'corenumber', 'avgshortestpath']
        classfier_column = ['riskfactor']

        X_train = self.train_data.as_matrix(columns)
        Y_train = self.train_data.as_matrix(classfier_column)

        X_test = self.test_data.as_matrix(columns)
        Y_test = self.test_data.as_matrix(classfier_column)
        self.randomforest(X_train, Y_train, X_test, Y_test)

    def randomforest(self, X_train, Y_train, X_test, Y_test):
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
        model = RandomForestClassifier()
        ######### Without GridSearch #####################
        model.fit(X_train, Y_train.ravel())
        y_true, y_pred = Y_test, model.predict(X_test)
        print "-----Random Forest without GridSearch-----"
        print classification_report(y_true, y_pred)
        ##################################################

        ########## With gridsearch #######################
        grid_values = {
            'n_estimators': [200, 700],
            #'max_features': ['auto', 'sqrt', 'log2'],
            'class_weight': ['balanced']
        }
        clf = GridSearchCV(RandomForestClassifier(), param_grid=grid_values, scoring="f1", cv=5)
        clf.fit(X_train, Y_train.ravel())
        y_true , y_pred = Y_test, clf.predict(X_test)
        print "-----Random Forest with GridSearch-----"
        #print clf.coef_
        print classification_report(y_true, y_pred)
        ##################################################

        ######### RFE ####################################
        params = clf.best_params_
        estimator = RandomForestClassifier(n_estimators=params['n_estimators'], class_weight=params['class_weight'])
        rfe = RFE(estimator, n_features_to_select=1, step=1)
        rfe = rfe.fit(X_train, Y_train.ravel())
        y_true, y_pred = Y_test, rfe.predict(X_test)
        features =['connectedComponents', 'triangles', 'coefficient', 'egonetSize', 'corenumber', 'avgshortestpath']
        sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), features))
        feature_selected = dict(zip(rfe.ranking_, features))
        result = [feature_selected[key] for key in sorted(feature_selected.keys())]

        ####################################################
        for numbers in range(len(result), 0, -1):
            X_train = self.train_data.as_matrix(result[:numbers])
            X_test = self.test_data.as_matrix(result[:numbers])
            estimator.fit(X_train, Y_train)
            y_true, y_pred = Y_test, estimator.predict(X_test)
            print "-----Random Forest-----"
            print "features - " + str(result[:numbers])
            print classification_report(y_true, y_pred)
        ##################################################