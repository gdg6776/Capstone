import sys
from sklearn.model_selection import train_test_split
from createhist import createhistogram
from decisionTree import decisiontreeclassifier
from egograph import ego
from logisticmodel import classifydata
from readData import read
from rfclassifier import randomforestclassifier


class mainGraphProgram(object):
    def __init__(self, graphname, riskfactor, name, data):
        self.graphname = graphname
        self.riskfactor = riskfactor
        self.name = name
        self.data = data

    def buildegonet(self):
        df = ""
        egraph = ego(self.graphname, self.riskfactor, self.name, self.data)
        return egraph.egodata()


class task(object):
    def __init__(self):
        pass

    def performtask(self, graph, riskfactor, name, nodes):
        index = 0

        print "Creating the egonet for each node and calculating core number, " \
              "triangle count, coefficients, egonetsize and connected components for " + str(name)
        mgp = mainGraphProgram(graph, riskfactor, name, nodes)
        egofeatures = mgp.buildegonet()
        egofeaturesnamelist = ['Connected Components ', 'Triangles ', 'Coefficients ', 'Egonet size ','Core Number size ']
        binvalues = [10,4,30,10,4]
        hist = createhistogram()
        print "Creating Histograms..."
        print ""


        for value in egofeaturesnamelist:
            hist.createGraph(egofeatures[index], "Value", "Probability", "Histogram for "+value+"{0}".format(name.upper()),
                             binvalues[index])
            index += 1


def main():
    graph = "dev.graphml"
    graph2 = "test.graphml"
    name = ""
    t = task()
    rd = read(graph)

    for value in range(0, 3):
        if value == 1:
            print "Performing analysis on users who are atRisk..."
            riskfactor = "1"
            name = "atrisk"
            graphdata= rd.readG()[0]
            t.performtask(graphdata, riskfactor, name, graphdata.nodes())
        elif value == 0:
            print "Performing analysis on users who are Not atRisk..."
            riskfactor = "0"
            name = "notatrisk"
            graphdata= rd.readG()[0]
            t.performtask(graphdata, riskfactor, name, graphdata.nodes())
        else:
            riskfactor = -1
            graphdata, dataframe = rd.readG()

            X = dataframe.as_matrix(['Node'])

            # Split the users into testing and training...
            x_train, x_test = train_test_split(X,test_size=0.3)

            # #Create a subgraph restricted to training users
            subG = graphdata.subgraph(x_train.ravel())

            #Compute graph features and create training data -  graph restricted to training users
            mgp = mainGraphProgram(subG, riskfactor, name, x_train.ravel())
            train_data = mgp.buildegonet()

            #Compute graph features and create testing data- graph restricted just to training + test (or dev) users only
            mgp = mainGraphProgram(graphdata, riskfactor, name, x_test.ravel())
            test_data = mgp.buildegonet()


            #Normalizing the test data...
            test_data['connectedComponents'] = (test_data['connectedComponents'] - test_data['connectedComponents'].mean()) / \
                                 test_data['connectedComponents'].std()

            test_data['triangles'] = (test_data['triangles'] - test_data[
                'triangles'].mean()) / \
                                               test_data['triangles'].std()

            test_data['coefficient'] = (test_data['coefficient'] - test_data[
                'coefficient'].mean()) / \
                                               test_data['coefficient'].std()

            test_data['egonetSize'] = (test_data['egonetSize'] - test_data[
                'egonetSize'].mean()) / \
                                               test_data['egonetSize'].std()

            test_data['corenumber'] = (test_data['corenumber'] - test_data[
                'corenumber'].mean()) / \
                                      test_data['corenumber'].std()

            print "Performing analysis on Egonet features using Logistic Regression..."
            # Logistic Model
            logistic = classifydata(train_data, test_data)
            logistic.classifier()

            # print "Performing analysis on Egonet features using Support Vector Machines..."
            # # SVM
            # svmmod = svmMod(train_data, test_data)
            # svmmod.model()

            print "Performing analysis on Egonet features using Random Forest..."
            # Random Forest
            rf = randomforestclassifier(train_data, test_data)
            rf.model()

            # Decision Tree
            print "Performing analysis on Egonet features using Decision Tree..."
            dt = decisiontreeclassifier(train_data, test_data)
            dt.model()

            # # mgp = mainGraphProgram(graphdata, riskfactor, name, graphdata.nodes())
            # # dataforcov = mgp.buildegonet()
            # # cov = covariance.empirical_covariance(dataforcov.as_matrix(), assume_centered=False)
            # # print cov

            sys.exit()

if __name__ == '__main__':
    main()
