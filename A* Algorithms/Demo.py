import os
import pickle
import sys
from BackwardA import bAgent
from AdaptiveA import aAgent
from ForwardA import fAgent
from Maze import Maze
from SmallGForward import sgAgent


class Demo:
    limit = 3  #change this to change the amount of mazes generated

    @staticmethod
    def generateMazeList():
        mazeList = list()
        for i in range(0, Demo.limit):
            mazeList.append(Maze())
        return mazeList

    @staticmethod
    def doForward(maze):
        agent = fAgent(maze.sx, maze.sy, maze.gx, maze.gy)
        agent.forwardA(mazeList[i])

    @staticmethod
    def doAdaptive(maze):
        agent = aAgent(maze.sx, maze.sy, maze.gx, maze.gy)
        agent.adaptiveA(mazeList[i])

    @staticmethod
    def doSmallGForward(maze):
        agent = sgAgent(maze.sx, maze.sy, maze.gx, maze.gy)
        agent.forwardA(mazeList[i])

    @staticmethod
    def doBackward(maze):
        agent = bAgent(maze.sx, maze.sy, maze.gx, maze.gy)
        agent.backwardA(maze)

    @staticmethod
    def importMazes():
        src = input("Please input a path to a pickle file containing a mazeList")
        importFile = open(src, 'rb')
        mazeList = pickle.load(importFile)
        importFile.close()
        return mazeList

    @staticmethod
    def export(mazeList):
        file = open(os.path.join(sys.path[0], 'mazeList.pkl'), 'wb')
        pickle.dump(mazeList, file)
        file.close()
        print("Saved to " + os.path.join(sys.path[0], 'mazeList.pkl'))


if __name__ == "__main__":
    response = input("Would you like to import a maze list? Y/N").lower()
    while response != 'y' and response != 'n':
        response = input("Would you like to import a maze? Y/N").lower()
    if response == 'y':
        mazeList = Demo.importMazes()
    else:
        mazeList = Demo.generateMazeList()
        print("mazeList successfully generated")
    sType = -1
    while sType != '1' and sType != '2' and sType != '3' and sType != '4':
        sType = input('Press 1 for forwardA with larger g comparison on ties (default)*\n' +
                      'Press 2 for forwardA with smaller g comparison on ties*\n' +
                      'Press 3 for adaptiveA* \n' +
                      'Press 4 for backwardA')
    for i in range(0, Demo.limit):
        if sType == '1':
            print("Performing forwardA* search (large g) on maze " + str(i))
            Demo.doForward(mazeList[i])
        elif sType == '2':
            print("Performing forwardA* search (small g) on maze " + str(i))
            Demo.doSmallGForward(mazeList[i])
        elif sType == '3':
            print("Performing adaptiveA* search on maze " + str(i))
            Demo.doAdaptive(mazeList[i])
        elif sType == '4':
            print("Performing backwardA* search on maze " + str(i))
            Demo.doBackward(mazeList[i])
        #input("Done: Input anything to continue")

    isExport = '69'
    while isExport != 'y' and isExport != 'n':
        isExport = input("Finished Demo: Would you like to export the mazeList? Y/N?").lower()
    if isExport == 'y':
        Demo.export(mazeList)
