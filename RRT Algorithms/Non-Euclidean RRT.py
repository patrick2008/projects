"""
COMS 4733 Fall 2021 Homework 3
Scaffolding code for building a RRT for RR arm motion planning
Portions accredited to the PythonRobotics project
https://pythonrobotics.readthedocs.io/en/latest/
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random

"""
Simulation environment
"""
LINK_LENGTH = [1, 1]    # lengths of arm links

# environment 1
# OBSTACLES = [[1.75, 0.75, 0.6], [-.5, 1.5, 0.5], [0, -1, 0.7]] # circular obstacles [x, y, r]
# START = (1.0, 0.0)      # start joint configuration
# GOAL = (-3.0, -1.0)     # goal joint configuration

# environment 2
OBSTACLES = [[1.75, 0.75, 0.6], [-.55, 1.5, 0.5], [0, -1, 0.7], [-2, -0.5, 0.6]]
START = (-3.0, 1.0)
GOAL = (-0.5, 0.5)

"""
RRT parameters
"""
MAX_NODES = 800     # maximum number of nodes to sample
BIAS = 0.05          # probability that a sampled node is manually set to the goal
DELTA = 0.08       # length of tree branches
EDGE_INC = 0.02      # edge sampling increment for collision detection


"""
Class for an n-link arm
"""
class NLinkArm(object):
    def __init__(self, link_lengths, joint_angles):
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = np.array(joint_angles)
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + \
                self.link_lengths[i - 1] * \
                np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + \
                self.link_lengths[i - 1] * \
                np.sin(np.sum(self.joint_angles[:i]))

        self.end_effector = np.array(self.points[self.n_links]).T


"""
Utility functions
"""
def detect_collision(arm, config):
    """
    :param arm: NLinkArm object
    :param config: Configuration (joint angles) of the arm
    :return: True if any part of arm collides with obstacles, False otherwise
    """
    arm.update_joints(config)
    points = arm.points
    for k in range(len(points) - 1):
        for circle in OBSTACLES:
            a_vec = np.array(points[k])
            b_vec = np.array(points[k+1])
            c_vec = np.array([circle[0], circle[1]])
            radius = circle[2]

            line_vec = b_vec - a_vec
            line_mag = np.linalg.norm(line_vec)
            circle_vec = c_vec - a_vec
            proj = circle_vec.dot(line_vec / line_mag)

            if proj <= 0:
                closest_point = a_vec
            elif proj >= line_mag:
                closest_point = b_vec
            else:
                closest_point = a_vec + line_vec * proj / line_mag

            if np.linalg.norm(closest_point - c_vec) <= radius:
                return True

    return False


def closest_euclidean(q, qp):
    """
    :param q, qp: Two 2D vectors in S1 x S1
    :return: qpp, dist. qpp is transformed version of qp so that L1 Euclidean distance between q and qpp
    is equal to toroidal distance between q and qp. dist is the corresponding distance.
    """
    q = np.array(q)
    qp = np.array(qp)

    A = np.meshgrid([-1,0,1], [-1,0,1])
    qpp_set = qp + 2*np.pi*np.array(A).T.reshape(-1,2)
    distances = np.linalg.norm(qpp_set-q, 1, axis=1)
    ind = np.argmin(distances)
    dist = np.min(distances)

    return qpp_set[ind], dist


def boundary(q1):
    q1 = list(q1)
    if q1[0] < -1*math.pi:
        q1[0] += 2*math.pi
    if q1[0] > math.pi:
        q1[0] += -2*math.pi
    if q1[1] < -1*math.pi:
        q1[1] += 2*math.pi
    if q1[1] > math.pi:
        q1[1] += -2*math.pi
    q1 = tuple(q1)
    return q1

"""
RRT construction and path planning
IMPLEMENT THESE FUNCTIONS
"""
def clear_path(arm, q1, q2):
    """
    :param arm: NLinkArm object
    :param q1, q2: Two configurations in S1 x S1
    :return: True if edge between q1 and q2 sampled at EDGE_INC increments collides with obstacles, False otherwise
    """
    while 1 ==1:
        qpp,dist = closest_euclidean(q1,q2)
        q2=qpp
        if dist < EDGE_INC:
            break
        v = [EDGE_INC*(q2[0]-q1[0])/dist, EDGE_INC*(q2[1]-q1[1])/dist]
        q1 = (q1[0]+v[0],q1[1]+v[1])
        q1 = boundary(q1)
        if detect_collision(arm,q1):
            return True
    return False
    




def find_qnew(tree, qrand):
    """
    :param tree: RRT dictionary {(node_x, node_y): (parent_x, parent_y)}
    :param qrand: Randomly sampled configuration
    :return: qnear in tree, qnew between qnear and qrand with distance DELTA from qnear
    """
    qnear = list(tree.keys())[0]
    for q in tree:
        qpp1,dist1 = closest_euclidean(q,qrand)
        qpp2,dist2 = closest_euclidean(qnear,qrand)
        if dist1 <= dist2:
            qnear = q
    qpp2,dist2 = closest_euclidean(qnear,qrand)
    qrand=qpp2
    d = dist2
    qnew = (qnear[0]+(DELTA*(qrand[0]-qnear[0])/d),qnear[1]+(DELTA*(qrand[1]-qnear[1])/d))
    qnew = boundary(qnew)
    qpp3,dist3 = closest_euclidean(qnear, qnew)
    qpp4,dist4 = closest_euclidean(qnear, qrand)
    if dist3 > dist4:
        qnew = qrand
    return qnear, (qnew[0],qnew[1])




def find_qnew_greedy(arm,tree, qrand):
    """
    :param arm: NLinkArm object
    :param tree: RRT dictionary {(node_x, node_y): (parent_x, parent_y)}
    :param qrand: Randomly sampled configuration
    :return: qnear in tree, qnew between qnear and qrand as close as possible to qrand in increments of DELTA
    """
    qnear = list(tree.keys())[0]
    for q in tree:
        qpp1,dist1 = closest_euclidean(q,qrand)
        qpp2,dist2 = closest_euclidean(qnear,qrand)
        if dist1 <= dist2:
            qnear = q
    qpp2,dist2 = closest_euclidean(qnear,qrand)
    qrand=qpp2
    d = dist2
    qnew = (qnear[0]+(DELTA*(qrand[0]-qnear[0])/d),qnear[1]+(DELTA*(qrand[1]-qnear[1])/d))
    qnew = boundary(qnew)
    qpp3,dist3 = closest_euclidean(qnear, qnew)
    qpp4,dist4 = closest_euclidean(qnear, qrand)
    if dist3 > dist4:
        qnew = qrand
    else:
        while not detect_collision(arm,qnew):
            qold = qnew
            qpp,dist = closest_euclidean(qnear,qrand)
            qrand=(qpp[0],qpp[1])
            d = dist
            qnew = (qnew[0]+(DELTA*(qrand[0]-qnew[0])/d),qnew[1]+(DELTA*(qrand[1]-qnew[1])/d))
            qnew = boundary(qnew)
            if detect_collision(arm,qnew):
                qnew = qold
                break
            qpp4,dist4 = closest_euclidean(qnew, qrand)
            if dist4 <=DELTA:
                qnew = qrand
                break
    return qnear, (qnew[0],qnew[1])



def construct_tree(arm):
    """
    :param arm: NLinkArm object
    :return: roadmap: Dictionary of nodes in the constructed tree {(node_x, node_y): (parent_x, parent_y)}
    :return: path: List of configurations traversed from start to goal
    """
    tree = {START:None}
    switch = 0
    for i in range(MAX_NODES):
        print(i)
        b = random.random()
        if BIAS >= b:
            qrand = GOAL
        else:
            qrand = (random.uniform(-1*math.pi,math.pi),random.uniform(-1*math.pi,math.pi))
        qnear,qnew = find_qnew(tree,qrand)
        while detect_collision(arm,qrand) or clear_path(arm,qnear,qrand):
            qrand = (random.uniform(-1*math.pi,math.pi),random.uniform(-1*math.pi,math.pi))
            qnear,qnew = find_qnew(tree,qrand)
        qpp, dist = closest_euclidean(qnew,GOAL)
        if dist < DELTA:
            qnew = GOAL
            switch = 1
            tree[qnew] = qnear
            break
        tree[qnew] = qnear

    if switch:
        path = [qnew]
        while tree[qnew]!=None:
            qnew = tree[qnew]
            path.append(qnew)
        return tree, list(reversed(path))
    else:
        return tree,[]

def construct_tree_greedy(arm):
    """
    :param arm: NLinkArm object
    :return: roadmap: Dictionary of nodes in the constructed tree {(node_x, node_y): (parent_x, parent_y)}
    :return: path: List of configurations traversed from start to goal
    """
    tree = {START:None}
    switch = 0
    for i in range(MAX_NODES):
        print(i)
        b = random.random()
        if BIAS >= b:
            qrand = GOAL
        else:
            qrand = (random.uniform(-1*math.pi,math.pi),random.uniform(-1*math.pi,math.pi))
        qnear,qnew = find_qnew_greedy(arm,tree,qrand)
        while detect_collision(arm,qrand) or clear_path(arm,qnear,qrand):
            qrand = (random.uniform(-1*math.pi,math.pi),random.uniform(-1*math.pi,math.pi))
            qnear,qnew = find_qnew_greedy(arm,tree,qrand)
        qpp, dist = closest_euclidean(qnew,GOAL)
        if dist < DELTA:
            qnew = GOAL
            switch = 1
            tree[qnew] = qnear
            break
        tree[qnew] = qnear

    if switch:
        path = [qnew]
        while tree[qnew]!=None:
            qnew = tree[qnew]
            path.append(qnew)
        return tree, list(reversed(path))
    else:
        return tree,[]

"""
Plotting and visualization functions
"""

def get_occupancy_grid(arm, M):
    grid = [[0 for _ in range(M)] for _ in range(M)]
    theta_list = [2 * i * np.pi / M for i in range(-M // 2, M // 2 + 1)]
    for i in range(M):
        for j in range(M):
            grid[i][j] = int(detect_collision(arm, [theta_list[i], theta_list[j]]))
    return np.array(grid)

def plot_roadmap(ax, roadmap):
    for node, parent in roadmap.items():
        if parent is not None:
            euc_parent, _ = closest_euclidean(node, parent)
            euc_node, _ = closest_euclidean(parent, node)
            ax.plot([node[0], euc_parent[0]], [node[1], euc_parent[1]], "-k")
            ax.plot([euc_node[0], parent[0]], [euc_node[1], parent[1]], "-k")
        ax.plot(node[0], node[1], ".b")

def plot_arm(plt, ax, arm):
    for obstacle in OBSTACLES:
        circle = plt.Circle((obstacle[0], obstacle[1]), radius=obstacle[2], fc='k')
        plt.gca().add_patch(circle)

    for i in range(arm.n_links + 1):
        if i is not arm.n_links:
            ax.plot([arm.points[i][0], arm.points[i + 1][0]], [arm.points[i][1], arm.points[i + 1][1]], 'r-')
        ax.plot(arm.points[i][0], arm.points[i][1], 'k.')

def visualize_spaces(arm):
    plt.subplots(1, 2)

    plt.subplot(1, 2, 1)
    grid = get_occupancy_grid(arm, 200)
    plt.imshow(np.flip(grid.T, axis=0))
    plt.xticks([0,50,100,150,200], ["-\u03C0", "-\u03C0/2", "0", "\u03C0/2", "\u03C0"])
    plt.yticks([0,50,100,150,200], ["\u03C0", "\u03C0/2", "0", "-\u03C0/2", "-\u03C0"])
    plt.title("Configuration space")
    plt.xlabel('joint 1')
    plt.ylabel('joint 2')

    ax = plt.subplot(1, 2, 2)
    arm.update_joints(START)
    plot_arm(plt, ax, arm)
    plt.title("Workspace")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('scaled')
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.show()

def animate(arm, roadmap, route):
    ax1 = plt.subplot(1, 2, 1)
    plot_roadmap(ax1, roadmap)
    if route:
        plt.plot(route[0][0], route[0][1], "Xc")
        plt.plot(route[-1][0], route[-1][1], "Xc")
    plt.title("Configuration space")
    plt.xlabel('joint 1')
    plt.ylabel('joint 2')
    plt.axis('scaled')
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)

    ax2 = plt.subplot(1, 2, 2)
    arm.update_joints(START)
    plot_arm(plt, ax2, arm)
    plt.title("Workspace")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('scaled')
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.pause(1)

    for config in route:
        arm.update_joints([config[0], config[1]])
        ax1.plot(config[0], config[1], "xr")
        ax2.lines = []
        plot_arm(plt, ax2, arm)
        # Uncomment here to save the sequence of frames
        #plt.savefig('frame{:04d}.png'.format(i))
        plt.pause(0.3)

    plt.show()


"""
Main function
"""
def main():
    ARM = NLinkArm(LINK_LENGTH, [0,0])
    #visualize_spaces(ARM)
    roadmap, route = construct_tree(ARM)
    if not route:
        print("No path found!")
    animate(ARM, roadmap, route)

if __name__ == '__main__':
    main()