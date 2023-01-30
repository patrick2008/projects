import os
from Scan import Scan
from Regression import Regression
from Gap import Gap
from ScaleDown import ScaleDown

class NaiveBayes:

    def __init__(self, samples, type):
        self.samples = samples
        self.cntY = {}

        self.countY()
        if type == 'digit' or type == 'd':
            self.buildPhi()
        else:
            self.buildPhiFace()
        self.nLabels = len(self.cntY)

    def countY(self):
        for node in self.samples:
            if node.label not in self.cntY:
                self.cntY[node.label] = 1
            else:
                self.cntY[node.label] += 1

    def p_feature(self, x, j, y):
        k = 0.01
        numY =self.cntY[y]
        domainSizeY = len(self.cntY)
        return (self.p_phiXandYTrue(x, j, y) + k) / (numY + k * domainSizeY)

    def p_featureFalse(self, x, j, y):
        k = 0.01
        numYFalse = self.complement(self.cntY[y])
        domainSizeY = len(self.cntY)
        return (self.p_phiXandYFalse(x, j, y) + k) / (numYFalse + k * domainSizeY)

    def p_xGivenYTrue(self, x, y):
        prod = 1.0
        for j in x.phiVector:
            prod *= self.p_feature(x, j, y)
        return prod

    def p_phiXandYTrue(self, x, j, y):
        count = 0
        for n in self.samples:
            if n.phiVector.get(j) == x.phiVector.get(j) and n.label == y:
                count += 1
        return count

    def p_xGivenYFalse(self, x, y):
        prod = 1.0
        for j in x.phiVector:
            prod *= self.p_featureFalse(x, j, y)  # p feature false is 0
        return prod

    def p_phiXandYFalse(self, x, j, y):
        count = 0
        for n in self.samples:
            if n.phiVector.get(j) == x.phiVector.get(j) and n.label != y:
                count += 1
        return count

    def complement(self, num):
        return len(self.samples) - float(num)

    def liklihoodRatio(self, x, y):
        top = self.p_xGivenYTrue(x, y) * self.cntY[y]
        bottom = self.p_xGivenYFalse(x, y) * self.complement(self.cntY[y])
        if bottom == 0:
            bottom = 0.00000001
        liklihoodRatio = top / bottom
        return liklihoodRatio

    def predict(self, x):
        max_p = 0
        max_label = None
        for l in self.cntY:
            p = self.liklihoodRatio(x, l)
            if p > max_p:
                max_p = p
                max_label = l
        return max_p, max_label

    def buildPhiFace(self):  # number of + and # over total pixels in line
        for node in self.samples:
            xList, yList = Regression.makeLists(node.image)
            m, b = Regression.findRegression(xList, yList)
            node.phiVector['m'] = round(m, 1)
            node.phiVector['hGap'] = round(Gap.horizontal(node.image))
            node.phiVector['vGap'] = round(Gap.vertical(node.image))

    def buildPhi(self):  # number of + and # over total pixels in line
        for node in self.samples:
            scaled = ScaleDown.scale(node.image, 0.25)
            i = 0
            scaledLines = scaled.split('\n')
            del scaledLines[-1]
            for line in scaledLines:
                node.phiVector['scaled' + str(i)] = line
                i += 1

    @staticmethod
    def nbFace(trainingRate):
        relpath = os.path.dirname(__file__)
        facex = os.path.join(relpath, r'data/facedata/facedatatrain')
        facey = os.path.join(relpath, r'data/facedata/facedatatrainlabels')
        faceHeight = 70
        instances = Scan.scanIn(facex, facey, faceHeight, trainingRate)
        bayes = NaiveBayes(instances, 'f')
        ftestx = os.path.join(relpath, r'data/facedata/facedatatest')
        ftesty = os.path.join(relpath, r'data/facedata/facedatatestlabels')
        testInstances = Scan.scanIn(ftestx, ftesty, faceHeight, trainingRate)
        testBayes = NaiveBayes(testInstances, 'f')  # assigns phivalues to all the test images
        print(bayes.cntY)
        total = 0
        correct = 0
        for x in testInstances:
            total += 1
            p, label = bayes.predict(x)
            if (x.label == label):
                correct += 1
        print(f"Percent Correct: {correct / total * 100}%")
        print('Printing Example...')
        print('Likelihood Ratio: ' + str(p))
        print('Feature Vector:')
        print(testInstances[-1].phiVector)
        print('Predicted Label: ' + label)
        print('Actual Label: ' + testInstances[-1].label)
        print('Actual Image:')
        print(testInstances[-1].image)

    @staticmethod
    def nbDigit(trainingRate):
        relpath = os.path.dirname(__file__)
        srcx = os.path.join(relpath, r'data/digitdata/trainingimages')
        srcy = os.path.join(relpath, r'data/digitdata/traininglabels')
        digitHeight = 28
        instances = Scan.scanIn(srcx, srcy, digitHeight, trainingRate)
        bayes = NaiveBayes(instances, 'd')
        srcTestX = os.path.join(relpath, r'data/digitdata/testimages')
        srcTestY = os.path.join(relpath, r'data/digitdata/testlabels')
        testInstances = Scan.scanIn(srcTestX, srcTestY, digitHeight, trainingRate)
        testBayes = NaiveBayes(testInstances, 'd')  # assigns psivalues to all the test images
        print(bayes.cntY)
        total = 0
        correct = 0
        for x in testInstances:
            total += 1
            p, label = bayes.predict(x)
            if (x.label == label):
                correct += 1
        print(f"Percent Correct: {correct / total * 100}%")
        print('Printing Example...')
        print('Likelihood Ratio: ' + str(p))
        print('Feature Vector:')
        print(testInstances[-1].phiVector)
        print(ScaleDown.scale(testInstances[-1].image, 0.25))
        print('Predicted Label: ' + label)
        print('Actual Label: ' + testInstances[-1].label)
        print('Actual Image:')
        print(testInstances[-1].image)


if __name__ == '__main__':
    rate = 1
    NaiveBayes.nbFace(rate)
    NaiveBayes.nbDigit(rate)
