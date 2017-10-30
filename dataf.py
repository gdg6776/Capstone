import numpy as np
import pandas as pd


class dataframe(object):
    def __init__(self):
        pass

    def createDataFrame(self, cc, tri, coeff, egoSize, hasrisk, corenumberlist, bridgelist ,nodesList):
        attributes = ['connectedComponents', 'triangles', 'coefficient', 'egonetSize', 'riskfactor', 'corenumber', 'bridgelist']

        new_data = np.array([cc, tri, coeff, egoSize, hasrisk, corenumberlist, bridgelist], dtype=float).transpose()

        df = pd.DataFrame(data=new_data, index=[i for i in nodesList],
                          columns=attributes)

        return df