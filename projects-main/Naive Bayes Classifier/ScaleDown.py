import math
import numpy as np

class ScaleDown:
    @staticmethod
    def scale(image, rate):
        splitImage = image.split('\n')
        del splitImage[-1]
        lineCount = len(splitImage)
        lineSize = len(splitImage[0])
        scaledLines = math.floor(lineCount * rate)
        scaledSize = math.floor(lineSize * rate)
        scaledImage = np.full((scaledLines, scaledSize + 1), '-1', np.dtype(str)) #+1 for the newline char
        for i in range(0, scaledLines):
            for j in range(0, scaledSize):
                filledCount = 0
                blankCount = 0
                for k in range(int(i / rate), int(i / rate + 1 / rate)):
                    for l in range(int(j / rate), int(j / rate + 1 / rate)):
                        if splitImage[k][l] != ' ':
                            filledCount += 1
                        else:
                            blankCount += 1
                if filledCount >= blankCount:
                    scaledImage[i][j] = '1'
                else:
                    scaledImage[i][j] = '0'
            scaledImage[i][j + 1] = '\n'
        return ScaleDown.toString(scaledImage)

    @staticmethod
    def toString(scaledImage):
        lines = ''
        for i in scaledImage:
            for j in i:
                lines += j
        return lines
