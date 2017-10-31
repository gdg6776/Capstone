import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import cumsum
from operator import itemgetter
import math


class createhistogram(object):
    def __init__(self):
        pass

    def createGraph(self, dict_1, x, y, title, range_value):
        counter = 0
        list_keys = []
        newDict = {}
        new_list = []
        binning = []
        # print title
        # print dict_1.values()
        # print " "


        myset = set(dict_1.values())
        binning = list(myset)
        print binning
        print ""


        # if title == 'Histogram for Coefficients ATRISK' or title == 'Histogram for Coefficients NOTATRISK':
        #     (counts, bins, patches) = plt.hist(dict_1.values(),bins=range(range_value) ,normed=True,
        #                                   cumulative=True)
        #
        #     plt.xlabel(x)
        #     plt.ylabel(y)
        #     plt.title(title)
        # else:
        (counts, bins, patches) = plt.hist(dict_1.values(),bins= range(range_value),normed=True,
                                      cumulative=True)

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)
        plt.show(title)