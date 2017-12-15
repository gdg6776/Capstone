import sys
from sklearn.model_selection import train_test_split
from createhist import createhistogram
from decisionTree import decisiontreeclassifier
from egograph import ego
from logisticmodel import classifydata
from readData import read
from rfclassifier import randomforestclassifier
from scipy.stats import rankdata
from svmModel import svmMod
import pdb

# CMH
# Not sure about the best place for this yet, so dumping it here
def normalize_data(data):
        for feature in ['connectedComponents', 'triangles', 'coefficient', 'egonetSize', 'corenumber']:
            data[feature] = rankdata(data[feature])/float(len(data))
        #data["riskfactor"] = 1 - data["riskfactor"]
        return data

 
class mainGraphProgram(object):
    def __init__(self, graphname, riskfactor, name, data):
        self.graphname = graphname
        self.riskfactor = riskfactor
        self.name = name
        self.data = data

    def buildegonet(self):
        df = ""
        egraph = ego(self.graphname, self.riskfactor, self.name, self.data)
        return_val = egraph.egodata()
        #pdb.set_trace()
        return return_val


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
    graph = sys.argv[1]
    graph2 = sys.argv[1]
    name = ""
    t = task()
    rd = read(graph)
    graphdata, dataframe = rd.readG()

    for value in range(0, 3):
        if value == 1:
            print "Performing analysis on users who are atRisk..."
            riskfactor = "1"
            name = "atrisk"
            t.performtask(graphdata, riskfactor, name, graphdata.nodes())
        elif value == 0:
            print "Performing analysis on users who are Not atRisk..."
            riskfactor = "0"
            name = "notatrisk"
            t.performtask(graphdata, riskfactor, name, graphdata.nodes())
        else:
            riskfactor = -1
            X = dataframe.as_matrix(['Node'])

            # Split the users into testing and training...
            x_train, x_test = train_test_split(dataframe,test_size=0.3)

            # #Create a subgraph restricted to training users
            subG = graphdata.subgraph(x_train['Node'].ravel())

            #Compute graph features and create training data -  graph restricted to training users
            mgp = mainGraphProgram(subG, riskfactor, name, x_train['Node'].ravel())
            train_data = mgp.buildegonet()
           
            train_data = normalize_data(train_data)

            #pdb.set_trace()
            #Compute graph features and create testing data- graph restricted just to training + test (or dev) users only
            mgp = mainGraphProgram(graphdata, riskfactor, name, dataframe['Node'].ravel())
            test_data = mgp.buildegonet()
            test_data = normalize_data(test_data)


            print "Performing analysis on Egonet features using Logistic Regression..."
            # Logistic Model
            logistic = classifydata(train_data, test_data.loc[x_test['Node']])
            logistic.classifier()

            print "Performing analysis on Egonet features using Support Vector Machines..."
            # SVM
            svmmod = svmMod(train_data, test_data.loc[x_test['Node']])
            svmmod.model()

            print "Performing analysis on Egonet features using Random Forest..."
            # Random Forest
            rf = randomforestclassifier(train_data, test_data.loc[x_test['Node']])
            rf.model()

            # Decision Tree
            print "Performing analysis on Egonet features using Decision Tree..."
            dt = decisiontreeclassifier(train_data, test_data.loc[x_test['Node']])
            dt.model()

            sys.exit()

if __name__ == '__main__':
    main()