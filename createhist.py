import matplotlib.pyplot as plt


class createhistogram(object):
    def __init__(self):
        pass

    def createGraph(self, dict_1, x, y, title, range_value):
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.hist(dict_1.values(), bins=range(range_value), normed=True,
                                           cumulative=True)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)
        fig.savefig(title + str(".pdf"), dpi = fig.dpi)
