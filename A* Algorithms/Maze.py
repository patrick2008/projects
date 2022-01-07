from Node import Node
import random
import numpy as np
import matplotlib.pyplot as plt


# from matplotlib import colors

class Maze:
    def __init__(self):
        w, h = 101, 101
        self.grid = [[Node(x, y) for x in range(w)] for y in range(h)]
        self.build_maze()
        self.setXY()
        self.setStart()
        self.setGoal()
        self.setH()
        self.resetVist()

    def resetVist(self):
        for i in range(0, 101):
            for j in range(0, 101):
                self.grid[i][j].isVisited = False

    def setXY(self):
        for i in range(0, 101):
            for j in range(0, 101):
                self.grid[i][j].x = i
                self.grid[i][j].y = j

    def build_maze(self):
        # 0,0 is in top left corner
        counter = 0
        stack = []
        while counter < 101 * 101:
            x = random.randint(0, 100)  # randint is inclusive lol
            y = random.randint(0, 100)
            mark = self.grid[x][y]
            if mark.isVisited:
                continue
            mark.isVisited = True
            counter += 1
            stack.append((x, y))
            while stack:  # while not empty
                x, y = stack[-1]  # gets coordinates of last in list
                if self.isDeadEnd(x, y):
                    # if all adjacent cells are blocked or out of bounds or visited
                    stack.pop()  # pops last in list
                else:
                    # print('hit')
                    r = random.randint(0, 3)
                    if r == 0:  # go north
                        try:
                            mark = self.grid[x][y - 1]
                            if mark.isVisited:
                                continue  # try again with same node
                            mark.isVisited = True
                            if random.random() > 0.7:
                                mark.isBlocked = True
                            else:
                                stack.append((x, y - 1))  # append mark or the coordinates?
                            counter += 1
                        except IndexError:  # catches grid edges
                            continue
                    elif r == 1:  # go east
                        try:
                            mark = self.grid[x + 1][y]
                            if mark.isVisited:
                                continue  # try again with same node
                            mark.isVisited = True
                            if random.random() > 0.7:
                                mark.isBlocked = True
                            else:
                                stack.append((x + 1, y))  # append mark or the coordinates?
                            counter += 1
                        except IndexError:  # catches grid edges
                            continue
                    elif r == 2:  # go south
                        try:
                            mark = self.grid[x][y + 1]
                            if mark.isVisited:
                                continue  # try again with same node
                            mark.isVisited = True
                            if random.random() > 0.7:
                                mark.isBlocked = True
                            else:
                                stack.append((x, y + 1))  # append mark or the coordinates?
                            counter += 1
                        except IndexError:  # catches grid edges
                            continue
                    else:  # go west
                        try:
                            mark = self.grid[x - 1][y]
                            if mark.isVisited:
                                continue  # try again with same node
                            mark.isVisited = True
                            if random.random() > 0.7:
                                mark.isBlocked = True
                            else:
                                stack.append((x - 1, y))  # append mark or the coordinates?
                            counter += 1
                        except IndexError:  # catches grid edges
                            continue

    def isDeadEnd(self, x, y):
        if ((y <= 0 or self.grid[x][y - 1].isBlocked or self.grid[x][y - 1].isVisited) and
                (y >= 100 or self.grid[x][y + 1].isBlocked or self.grid[x][y + 1].isVisited) and
                (x <= 0 or self.grid[x - 1][y].isBlocked or self.grid[x - 1][y].isVisited) and
                (x >= 100 or self.grid[x + 1][y].isBlocked or self.grid[x + 1][y].isVisited)):
            return True
        else:
            return False

    def setStart(self):
        while True:
            self.sx, self.sy = random.randint(0, 100), random.randint(0, 100)
            if not self.grid[self.sx][self.sy].isBlocked:
                self.start = self.grid[self.sx][self.sy]
                return self.start

    def setGoal(self):
        while True:
            self.gx, self.gy = random.randint(0, 100), random.randint(0, 100)
            if not self.grid[self.gx][self.gy].isBlocked:
                self.goal = self.grid[self.gx][self.gy]
                return self.goal

    def setH(self):
        for i in range(101):
            for j in range(101):
                self.grid[i][j].h = abs(self.gx - i) + abs(self.gy - j)

    def print(self):
        for x in range(101):
            for y in range(101):
                # print('[{}, {}]'.format(self.grid[x][y].g, self.grid[x][y].h), end = '')
                if self.grid[x][y].isBlocked == True:
                    print('[x]', end='')
                else:
                    print('[%d]' % self.grid[x][y].h, end='')
            print('\n')

    def pprint(self, ZZ, close=False):
        ax = plt.figure()
        Z = np.random.choice([0, 35, 25, 50, 10], size=[101, 101],
                             p=[1, 0, 0, 0, 0])  # 0=unblocked, 35=start, 25=path, 50= goal, 10= blocked
        # plt.axis([-1, 101, -1, 101])
        for x in range(101):
            for y in range(101):
                # print('[{}, {}]'.format(self.grid[x][y].g, self.grid[x][y].h), end = '')
                if self.grid[x][y].isBlocked:
                    Z[x][y] = 10
                    # plt.text(x,y,"x", fontsize = 8)
                elif ZZ[x][y] == 1:
                    Z[x][y] = 25
                # plt.text(x,y,"o", fontsize = 8)
            # print('\n')

        Z[self.sx][self.sy] = 35
        Z[self.gx][self.gy] = 50
        plt.imshow(Z, interpolation='nearest')
        plt.xticks([]), plt.yticks([])
        # plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
        if close:
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    i = 0
    while i < 1:
        m = Maze()
        m.print()
        i += 1
