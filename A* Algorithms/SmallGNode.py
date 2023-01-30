class SmallGNode:
    def __init__(self, x, y, f=0):
        self.isVisited = False
        self.g = 0
        self.h = 0
        self.isBlocked = False
        self.x, self.y = -1, -1
        self.searchVal = 0
        self.treePointer = None

    @property
    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f < other.f or(self.f == other.f and self.g < other.g)
        #return self.f < other.f

    def __repr__(self):
        return '({}, {}, f:{})'.format(self.x, self.y, self.f)
