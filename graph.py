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


def main():
    filtervalue = False
    graph = "dev.graphml"
    number = int(input("Enter 1 for atrisk and 0 for notatrisk for histograms or 2 for different model "))
    riskfactor = -1
    name = ""
    if number == 1:
        riskfactor = "1"
        name = "atrisk"
    elif number == 0:
        riskfactor = "0"
        name = "notatrisk"
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

        # #Logistic Model
        logistic = classifydata(train_data)
        logistic.classifier()

        # SVM
        svmmod = svmMod(train_data)
        svmmod.model()

        #Random Forest
        rf = randomforestclassifier(train_data)
        rf.model()

        # Decision Tree
        dt = decisiontreeclassifier(train_data)
        dt.model()
        sys.exit()


    rd = read(graph)
    mgp = mainGraphProgram(rd.readG(), riskfactor, name, filtervalue)

    if isinstance(mgp.buildegonet(), tuple):
        (connectedcomponents, triangles, coefficient, egonetSize, corenumber) = mgp.buildegonet()
        hist = createhistogram()
        hist.createGraph(connectedcomponents, "Value", "Frequency",
                         "Histogram for Connected Components {0}".format(name.upper()))
        hist.createGraph(triangles, "Value", "Frequency", "Histogram for Triangles {0}".format(name.upper()))
        hist.createGraph(coefficient, "Value", "Frequency", "Histogram for Coefficients {0}".format(name.upper()))
        hist.createGraph(egonetSize, "Value", "Frequency", "Histogram for Egonet size {0}".format(name.upper()))
        hist.createGraph(corenumber, "Value", "Frequency", "Histogram for Core Number size {0}".format(name.upper()))
    else:
        dataframe = mgp.buildegonet()
        print dataframe


if __name__ == '__main__':
    main()
