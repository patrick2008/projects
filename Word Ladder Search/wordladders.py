import enchant, string
import matplotlib.pyplot as plt
from heapq import heappush, heappop

def successors(state):
    """
    Given a word, find all possible English word results from changing one letter.
    Return a list of (action, word) pairs, where action is the index of the
    changed letter.
    """
    d = enchant.Dict("en_US")
    child_states = []
    for i in range(len(state)):
        new = [state[:i]+x+state[i+1:] for x in string.ascii_lowercase]
        words = [x for x in new if d.check(x) and x != state]
        child_states = child_states + [(i, word) for word in words]
    return child_states

def numDiff(s1,s2):
    diff = 0
    for i in range(min(len(s1),len(s2))):
        if s1[i]!=s2[i]:
            diff+=1
    return diff

"""
5.1: Best-first search
"""
def best_first_search(start, goal, f):
    """
    Inputs: Start state, goal state, priority function
    Returns node containing goal or None if no goal found, total nodes expanded,
    frontier size per iteration
    """
    node = {'state':start, 'parent':None, 'cost':0}
    frontier = []
    reached = {}
    nodes_expanded = 0
    frontier_size = [len(frontier)]
    goal_node = None
    heappush(frontier,(0, start, node))
    while len(frontier)>0:
        nodes_expanded+=1
        current = heappop(frontier)[2]
        frontier_size.append(len(frontier))
        s = current['state']
        if s== goal:
            goal_node = current
            break
        cost = current['cost']
        nextNodes = successors(current['state'])
        for i in range(len(nextNodes)):
            temp = nextNodes[i]
            temp = temp[1]
            if temp in reached:
                continue
            reached[temp] = 1
            diff = numDiff(s,temp)
            
            priority = f(current,goal)
            heappush(frontier, (priority, temp, {'state':temp, 'parent':current, 'cost':cost+1}))

    return goal_node, nodes_expanded, frontier_size


"""
5.2: Priority functions
"""
def f_dfs(node, goal=None):
    return 2000000000 - (node['cost']+1)

def f_bfs(node, goal=None):
    return node['cost']+1

def f_ucs(node, goal=None):
    return node['cost']+1

def f_astar(node, goal):
    diff = numDiff(node['state'], goal)#Hamming distance
    return node['cost']+1 + diff


def sequence(node):
    """
    Given a node, follow its parents back to the start state.
    Return sequence of words from start to goal.
    """
    words = [node['state']]
    while node['parent'] is not None:
        node = node['parent']
        words.insert(0, node['state'])
    return words


def plotFrontier(bfs, ucs, astar):
    plt.plot(bfs)
    plt.plot(ucs)
    plt.plot(astar)
    plt.legend(['BFS', 'UCS', 'A*'])
    plt.show()

if __name__ == '__main__':
    start ='small'#'cold' #'small'#'fat'
    goal = 'large'#'warm'#'large'#'cop'
    solution1 = best_first_search(start, goal, f_bfs)
    solution2 = best_first_search(start, goal, f_ucs)
    solution3 = best_first_search(start, goal, f_astar)
    plotFrontier(solution1[2],solution2[2],solution3[2])
