import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import cumsum
from operator import itemgetter

class createhistogram(object):
    def __init__(self):
        pass

    def createGraph(self, dict_1, x, y, title):
        counter = 0
        list_keys = []
        newDict = {}
        new_list = []


        for data in set(dict_1.values()):
            for key, value in dict_1.items():
                if value == data:
                    list_keys.append(key)
            newDict[data] = len(list_keys)
            list_keys = []

        total = sum(newDict.values())
        new_list = [(float(value) / total)*100 for value in newDict.values()]
        # print newDict


        if title == 'Histogram for Coefficients ATRISK' or title == 'Histogram for Coefficients NOTATRISK':
            plt.hist(newDict.keys(), bins=20, weights=newDict.values())
            plt.xlim(0, 2)
            plt.ylim(0, 100)
            plt.xlabel(x)
            plt.ylabel("Frequency")
            plt.title(title)
        else:
            plt.hist(newDict.keys(), weights=[(float(value) / total) * 100 for value in newDict.values()],
                     bins=range(len(newDict.keys())))
            plt.xlim(0, 10)
            plt.ylim(0, 100)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(title)
        plt.show()