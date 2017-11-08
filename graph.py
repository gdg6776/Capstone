import sys
from createhist import createhistogram
from decisionTree import decisiontreeclassifier
from egograph import ego
from logisticmodel import classifydata
from readData import read
from rfclassifier import randomforestclassifier
from svmModel import svmMod
from sklearn import cross_validation


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
    riskfactor = -1
    name = ""
    t = task()

    for value in range(0, 3):
        if value == 1:
            # print "Performing analysis on users who are atRisk..."
            # riskfactor = "1"
            # name = "atrisk"
            # rd = read(graph)
            # graphdata= rd.readG()[0]
            # t.performtask(graphdata, riskfactor, name, graphdata.nodes())
            pass

        elif value == 0:
            # print "Performing analysis on users who are Not atRisk..."
            # riskfactor = "0"
            # name = "notatrisk"
            # rd = read(graph)
            # graphdata= rd.readG()[0]
            # t.performtask(graphdata, riskfactor, name, graphdata.nodes())
            pass

        else:
            rd = read(graph)
            graphdata, dataframe = rd.readG()

            X = dataframe.as_matrix(['Node'])
            Y = dataframe.as_matrix(['Risk Factor'])

            # Split the data into testing and training...
            x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y)

            #Create a subgraph restricted to training users
            subG = graphdata.subgraph(x_train.ravel())

            #Compute graph features and create training data -  graph restricted to training users
            mgp = mainGraphProgram(subG, riskfactor, name, x_train.ravel())
            train_data = mgp.buildegonet()

            #Compute graph features and create testing data- graph restricted just to training + test (or dev) users only
            mgp = mainGraphProgram(graphdata, riskfactor, name, x_test.ravel())
            test_data = mgp.buildegonet()

            print "Performing analysis on Egonet features using Logistic Regression..."
            #Logistic Model
            logistic = classifydata(train_data, test_data)
            logistic.classifier()

            print "Performing analysis on Egonet features using Support Vector Machines..."
            # SVM
            svmmod = svmMod(train_data, test_data)
            svmmod.model()
            #
            print "Performing analysis on Egonet features using Random Forest..."
            # Random Forest
            rf = randomforestclassifier(train_data, test_data)
            rf.model()
            #
            # # Decision Tree
            print "Performing analysis on Egonet features using Decision Tree..."
            dt = decisiontreeclassifier(train_data, test_data)
            dt.model()
            sys.exit()


if __name__ == '__main__':
    main()
