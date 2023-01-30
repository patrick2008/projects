import numpy, copy

class Gap:

    @staticmethod
    def horizontal(image):
        splitImage = image.split('\n')
        del splitImage[-1]
        gapList = list()
        for i in splitImage:
            if i.strip() == '':
                continue
            flag = False
            count = 0
            filledCount = 0
            prev = None
            gapList.append(0)
            for c in i:
                if c == ' ' and flag is False:
                    if prev != ' ' and prev is not None:
                        count += 1
                        flag = True
                elif c == ' ' and flag is True:
                    count += 1
                elif c != ' ' and flag is False:
                    filledCount += 1
                    prev = c
                    continue
                elif c != ' ' and flag is True:
                    filledCount += 1
                    flag = False
                    gapList[-1] = count
                prev = c
            gapList[-1] = gapList[-1] / filledCount
        return numpy.mean(gapList)

    @staticmethod
    def vertical(image):
        splitImage = image.split('\n')
        del splitImage[-1]
        lineCount = len(splitImage)
        lineSize = len(splitImage[0])
        gapList = list()
        for c in range(0, lineSize):
            if Gap.columnBlank(splitImage, c, lineCount):
                continue
            flag = False
            count = 0
            filledCount = 0
            prev = None
            gapList.append(0)
            for i in range(0, lineCount):
                if splitImage[i][c] == ' ' and flag is False:
                    if prev != ' ' and prev is not None:
                        count += 1
                        flag = True
                elif splitImage[i][c] == ' ' and flag is True:
                    count += 1
                elif splitImage[i][c] != ' ' and flag is False:
                    prev = splitImage[i][c]
                    filledCount += 1
                    continue
                elif splitImage[i][c] != ' ' and flag is True:
                    flag = False
                    filledCount += 1
                    gapList[-1] = count
                prev = splitImage[i][c]
            gapList[-1] = gapList[-1] / filledCount
        return numpy.mean(gapList)

    @staticmethod
    def columnBlank(image, column, colSize):
        flag = True
        for i in range(0, colSize):
            if image[i][column] != ' ':
                flag = False
                return flag
        return flag