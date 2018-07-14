
# now can not work with tensorflow


import matplotlib.pyplot as plt

class CNN_Plot(object):

    def __init__(self):
        self.TrainData = []
        self.TestData = []
        self.Len = 0

    def AddData(self, TrainValue, TestValue):
        self.TrainData.append(TrainValue)
        self.TestData.append(TestValue)
        self.Len += 1

    def ShowPlot(self, Step=100):
        if self.Len == 0:
            return

        x = range(self.Len * Step)

        plt.plot(x, self.TrainData, label="train")
        plt.plot(x, self.TestData, label="test")
        plt.legend()
        plt.xlabel('times')
        plt.ylabel("accurate")

        plt.show()
