import sys
from createhist import createhistogram
from decisionTree import decisiontreeclassifier
from egograph import ego
from logisticmodel import ProcessingData
from logisticmodel import classifydata
from readData import read
from rfclassifier import randomforestclassifier
from svmModel import svmMod


class mainGraphProgram(object):
    def __init__(self, graphname, riskfactor, name, filtervalue):
        self.graphname = graphname
        self.riskfactor = riskfactor
        self.name = name
        self.filtervalue = filtervalue

    def buildegonet(self):
        df = ""
        egraph = ego(self.graphname, self.riskfactor, self.name, self.filtervalue)
        if isinstance(egraph.egodata(), tuple):
            return egraph.egodata()
        else:
            return egraph.egodata()


class task(object):
    def __init__(self):
        pass

    def performtask(self, graph, riskfactor, name, filtervalue):
        rd = read(graph)
        print "Creating the egonet for each node and calculating core number, " \
              "triangle count, coefficients, egonetsize and connected components for " + str(name)
        mgp = mainGraphProgram(rd.readG(), riskfactor, name, filtervalue)

        if isinstance(mgp.buildegonet(), tuple):
            (connectedcomponents, triangles, coefficient, egonetSize, corenumber) = mgp.buildegonet()
            hist = createhistogram()
            print "Creating Histograms..."
            print ""
            hist.createGraph(connectedcomponents, "Value", "Probability",
                             "Histogram for Connected Components {0}".format(name.upper()), 10)
            hist.createGraph(triangles, "Value", "Probability", "Histogram for Triangles {0}".format(name.upper()), 4)
            hist.createGraph(coefficient, "Value", "Probability", "Histogram for Coefficients {0}".format(name.upper()),
                             30)
            hist.createGraph(egonetSize, "Value", "Probability", "Histogram for Egonet size {0}".format(name.upper()),
                             10)
            hist.createGraph(corenumber, "Value", "Probability",
                             "Histogram for Core Number size {0}".format(name.upper()), 4)
        else:
            dataframe = mgp.buildegonet()
            print dataframe


def main():
    filtervalue = False
    graph = "dev.graphml"
    riskfactor = -1
    name = ""
    t = task()

    for value in range(0, 3):
        if value == 1:
            print "Performing analysis on users who are atRisk..."
            riskfactor = "1"
            name = "atrisk"
            t.performtask(graph, riskfactor, name, filtervalue)
        elif value == 0:
            riskfactor = "0"
            name = "notatrisk"
            print "Performing analysis on users who are Not atRisk..."
            t.performtask(graph, riskfactor, name, filtervalue)
        else:
            filtervalue = True
            graph1 = "dev.graphml"
            graph2 = "test.graphml"

            rd = read(graph1)
            mgp = mainGraphProgram(rd.readG(), riskfactor, name, filtervalue)
            dataframe_train = mgp.buildegonet()
            pd = ProcessingData(dataframe_train)
            train_data = pd.datapreprocessing()

            # rd = read(graph2)
            # mgp = mainGraphProgram(rd.readG(), riskfactor, name, filtervalue)
            # dataframe_test = mgp.buildegonet()
            # pd = ProcessingData(dataframe_test)
            # test_data = pd.datapreprocessing()

            print "Performing analysis on Egonet features using Logistic Regression..."
            # #Logistic Model
            logistic = classifydata(train_data)
            logistic.classifier()

            print "Performing analysis on Egonet features using Support Vector Machines..."
            # SVM
            svmmod = svmMod(train_data)
            svmmod.model()

            print "Performing analysis on Egonet features using Random Forest..."
            # Random Forest
            rf = randomforestclassifier(train_data)
            rf.model()

            # Decision Tree
            print "Performing analysis on Egonet features using Decision Tree..."
            dt = decisiontreeclassifier(train_data)
            dt.model()
            sys.exit()


if __name__ == '__main__':
    main()
