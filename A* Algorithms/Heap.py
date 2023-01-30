from Node import Node


class Heap:  # MinHeap
    def __init__(self):
        self.heap = list()

    def insert(self, val):
        self.heap.append(val)
        self.heapifyUp(len(self.heap) - 1)

    def clear(self):
        self.heap.clear()

    def heapifyUp(self, i):
        p = self.parent(i)
        while i > 0 and self.heap[i] < self.heap[p]:
            self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
            i = p
            p = self.parent(i)

    def heapifyDown(self):
        i = 0
        l = len(self.heap)
        while i < l // 2:
            left = self.left(i)
            right = self.right(i)
            which = right if right < l and self.heap[right] < self.heap[left] else left
            if self.heap[i] > self.heap[which]:
                self.heap[i], self.heap[which] = self.heap[which], self.heap[i]
                i = which
            else:
                break

    def extractMin(self):
        if len(self.heap) <= 0:
            return None
        v = self.heap[0]
        self.heap[0] = self.heap[-1]
        del self.heap[-1]
        self.heapifyDown()
        return v

    def getMin(self):
        if len(self.heap) <= 0:
            return None
        return self.heap[0]

    def deleteVal(self, val):
        for i in range(0, len(self.heap) - 1):
            if self.heap[i] == val:
                break
        v = self.heap[i]
        self.heap[i] = self.heap[-1]
        del self.heap[-1]
        self.heapifyDown()
        return v

    def isEmpty(self):
        return len(self.heap) == 0

    @staticmethod
    def parent(l):
        return (l - 1) // 2

    @staticmethod
    def left(i):
        return 2 * i + 1

    @staticmethod
    def right(i):
        return 2 * i + 2

    def __repr__(self):
        return str(self.heap)

    def __contains__(self, key):
        return key in self.heap


if __name__ == '__main__':
    h = Heap()
    h.insert(2)
    h.insert(4)
    h.insert(25)
    h.insert(12)
    h.insert(3)
    h.insert(-1)
    h.insert(0)
    print(h)
    print(h.extractMin())
    print(h.extractMin())
    h.insert(69)
    h.insert(25)
    print(h.extractMin())
    print(h)

    h2 = Heap()
    h2.insert(Node(2))
    h2.insert(Node(4))
    h2.insert(Node(4))
    s = Node(34)
    h2.insert(s)
    h2.insert(Node(12))
    h2.insert(Node(3))
    print(h2)
    print(h2.deleteVal(s))
    h2.clear()
    print(h2)
