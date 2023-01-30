from Heap import Heap
from Maze import Maze
from Node import Node
import math, sys
import numpy as np


class bAgent:
    W = 101
    H = 101

    def __init__(self, x, y, gx, gy):
        self.x = x
        self.y = y
        self.gx = gx
        self.gy = gy
        self.grid = [[Node(x, y) for x in range(self.W)] for y in range(self.H)]
        for i in range(self.W):
            for j in range(self.H):
                self.grid[i][j].h = abs(self.x - i) + abs(self.y - j)#compute from agent
                self.grid[i][j].x = i
                self.grid[i][j].y = j

    def node(self, x, y):
        return self.grid[x][y]

    def startNode(self):
        return self.node(self.x, self.y)

    def moveTo(self, node):
        self.x = node.x
        self.y = node.y

    def succ(self, s, a):
        if a == 'U':
            if s.y <= 0 or self.grid[s.x][s.y - 1].isBlocked:
                return None
            else:
                return self.grid[s.x][s.y - 1]
        elif a == 'R':
            if s.x >= self.W - 1 or self.grid[s.x + 1][s.y].isBlocked:
                return None
            else:
                return self.grid[s.x + 1][s.y]
        elif a == 'D':
            if s.y >= self.H - 1 or self.grid[s.x][s.y + 1].isBlocked:
                return None
            else:
                return self.grid[s.x][s.y + 1]
        elif a == 'L':
            if s.x <= 0 or self.grid[s.x - 1][s.y].isBlocked:
                return None
            else:
                return self.grid[s.x - 1][s.y]

    def computePath(self, OPEN, CLOSED, goal, counter, start):
        # each time you move you want to recalculate g and therefore f for neighboring cells
        while not OPEN.isEmpty() and start.g > OPEN.getMin().f:
            s = OPEN.extractMin()
            CLOSED.append(s)
            for a in ['U', 'R', 'D', 'L']:
                s1 = self.succ(s,a)
                if s1 is None:
                    continue
                if s1.searchVal < counter:
                    s1.g = math.inf
                    s1.searchVal = counter
                if s1.g > s.g + 1:
                    s1.g = s.g + 1
                    s1.treePointer = s
                    #print(s.x + s.y)
                    if s1 in OPEN:
                        OPEN.deleteVal(s1)
                    OPEN.insert(s1)

    def reversePath(self, start, goal):
        s = start
        path = []
        while s is not None and not (s.x == goal.x and s.y == goal.y):
            path.insert(0, s)
            prev = s
            s = s.treePointer
            prev.treePointer = None
        path.insert(0,goal)
        path =  path[::-1]
        return path

    def lookAround(self, s, m):
        for a in ['U', 'L', 'D', 'R']:
            z = self.succ(s, a)
            if z is not None:
                z.isBlocked = m.grid[z.x][z.y].isBlocked

    def backwardA(self, m):
        counter = 0
        ZZ = np.random.choice([0, 1], size=[101, 101], p=[1, 0])  # 0 = not traveled, 1=traveled
        CLOSED = list()
        OPEN = Heap()
        start = self.startNode()
        goal = self.node(self.gx, self.gy)
        self.lookAround(start, m)
        while self.x != goal.x or self.y != goal.y:
            counter += 1
            start.g = math.inf
            start.searchVal = counter
            goal.g = 0
            goal.searchVal = counter
            OPEN.clear()
            CLOSED.clear()
            OPEN.insert(goal)
            self.computePath(OPEN, CLOSED, goal, counter,start)
            if OPEN.isEmpty():  # open list is empty
                print("Cannot reach target")
                m.pprint(ZZ)
                sys.exit(0)
            path = self.reversePath(start, goal)
            zero = path[0]
            self.moveTo(zero)
            ZZ[zero.x][zero.y] = 1
            self.lookAround(zero, m)
            last = None
            for s in path[1:]:  # moves agent, goes in proper order and checks for isBlocked
                if not s.isBlocked:
                    self.moveTo(s)                 
                    ZZ[s.x][s.y] = 1
                    m.pprint(ZZ, close=True)
                    self.lookAround(s, m)
                    last = s
                else:
                    break
            if last is not None and goal.x != last.x and goal.y != last.y:
                start = last
        print("target reached")
        m.pprint(ZZ)


if __name__ == "__main__":
    m = Maze()
    agent = bAgent(m.sx, m.sy, m.gx, m.gy)
    agent.backwardA(m)
