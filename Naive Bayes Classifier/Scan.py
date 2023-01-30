import math, random
import numpy as np
from Node import Node
from itertools import islice


class Scan:

    @staticmethod
    def scanIn(srcx, srcy, height, rate):
        pydata = list()
        with open(str(srcx), 'r') as fx:
            while True:
                image = list(islice(fx, height))
                if not image: #EOF
                    break
                image = ''.join(image)
                pydata.append(image)
        fx.close()
        data = np.asarray(pydata)
        fy = open(str(srcy), 'r')
        pylabels = list()
        for line in fy:
            pylabels.append(line.strip())
        fy.close()
        labels = np.asarray(pylabels)
        if rate > 1: # for testing purposes
            instances = list()
            for i in range(0, len(labels)):
                instances.append(Node(data[i], labels[i]))
            npInstances = np.asarray(instances)
            return npInstances
        else:
            return Scan.randomSelect(data, labels, rate)

    @staticmethod
    def emptyLine(line):
        return line.strip() == ''

    @staticmethod
    def sanitize(data):
        for s in data:
            if len(s.split('\n')) < 7:  # 7-10 is a valid range for digit training
                data.remove(s)

    @staticmethod
    def randomSelect(data, labels, rate):  # insert number between 0 and 1
        lim = math.ceil(len(labels) * rate)
        pysamples = list()
        for i in range(0, lim):
            randnum = random.randint(0, len(labels) - 1)
            pysamples.append((Node(data[randnum], labels[randnum])))
        samples = np.asarray(pysamples)
        return samples