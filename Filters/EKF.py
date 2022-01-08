"""
COMS 4733 Fall 2021 Homework 4
Scaffolding code for localization using an extended Kalman filter
Inspired by a similar example on the PythonRobotics project
https://pythonrobotics.readthedocs.io/en/latest/
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


# "True" robot noise (filters do NOT know these)
WHEEL1_NOISE = 0.05
WHEEL2_NOISE = 0.1
BEARING_SENSOR_NOISE = np.deg2rad(1.0)

# Physical robot parameters (filters do know these)
RHO = 1
L = 1
MAX_RANGE = 18.0    # maximum observation range

# RFID positions [x, y]
RFID = np.array([[-5.0, -5.0],
                 [10.0, 0.0],
                 [10.0, 10.0],
                 [0.0, 15.0],
                 [-5.0, 20.0]])

# Covariances used by the estimators
Q = np.diag([0.1, 0.1, np.deg2rad(1.0)]) ** 2
R = np.diag([0.4, np.deg2rad(1.0)]) ** 2

# Other parameters
DT = 0.1            # time interval [s]
SIM_TIME = 30.0     # simulation time [s]

# Plot limits
XLIM = [-20,20]
YLIM = [-10,30]
show_animation = True


"""
Robot physics
"""
def input(time, x):
    # Control inputs to the robot at a given time for a given state
    psi1dot = 3.7
    psi2dot = 4.0
    return np.array([psi1dot, psi2dot])

def move(x, u):
    # Physical motion model of the robot: x_k = f(x_{k-1}, u_k)
    # Incorporates imperfections in the wheels
    theta = x[2]
    psi1dot = u[0] * (1 + np.random.rand() * WHEEL1_NOISE)
    psi2dot = u[1] * (1 + np.random.rand() * WHEEL2_NOISE)

    velocity = np.array([RHO/2 * np.cos(theta) * (psi1dot+psi2dot),
                         RHO/2 * np.sin(theta) * (psi1dot+psi2dot),
                         RHO/L * (psi2dot - psi1dot)])

    return x + DT * velocity

def measure(x):
    # Physical measurement model of the robot: z_k = h(x_k)
    # Incorporates imperfections in both range and bearing sensors
    z = np.zeros((0, 3))

    for i in range(len(RFID[:, 0])):
        dx = RFID[i, 0] - x[0]
        dy = RFID[i, 1] - x[1]
        r = math.sqrt(dx**2 + dy**2)
        phi = math.atan2(dy, dx) - x[2]

        if r <= MAX_RANGE:
            zi = np.array([[np.round(r),
                            phi + np.random.randn() * BEARING_SENSOR_NOISE,
                            i]])
            z = np.vstack((z, zi))

    return z


"""
Extended Kalman filter procedure
"""

def EKF(x, P, u, z):
    x, P = predict(x, P, u)
    x, P = update(x, P, z)
    return x, P


def predict(x, P, u):
    """
    :param x: State mean (x,y,theta) [size 3 array]
    :param P: State covariance [3x3 array]
    :param u: Robot inputs (u1,u2) [size 2 array]
    :return: Predicted state mean and covariance x and P
    """
    u1 = u[0]
    u2 = u[1]
    w_temp = np.random.normal(0, Q)
    w_k = np.array([w_temp[0][0],w_temp[1][1],w_temp[2][2]])
    x_k = [x[0]+DT*(RHO/2)*(math.cos(x[2]))*(u1+u2),x[1]+DT*(RHO/2)*(math.sin(x[2]))*(u1+u2),x[2]+DT*(RHO/L)*(u2-u1)]
    x_k+=w_k
    F_k = np.array([[1,0,-1*DT*(RHO/2)*math.sin(x[2])*(u1+u2)],[0,1,DT*(RHO/2)*math.cos(x[2])*(u1+u2)],[0,0,1]])
    x = np.array(x_k)
    P_k = np.matmul(np.matmul(F_k,P),np.transpose(F_k)) + Q
    P = P_k
    return x, P

def update(x, P, z):
    """
    :param x: State mean (x,y,theta) [size 3 array]
    :param P: State covariance [3x3 array]
    :param z: Sensor measurements [px3 array]. Each row contains range, bearing, and landmark's true (x,y) location.
    :return: Updated state mean and covariance x and P
    """
    if len(z)==0:
        return x,P
    p_size = z.shape[0]
    R_k = np.zeros((2*p_size,2*p_size))
    for i in range(0,len(R_k),2):
        R_k[i:i+2,i:i+2] = R

    v_k = np.random.multivariate_normal([0,0], R)

    H_k = []
    h_k = []
    for i in range(len(z)):
        pos = int(z[i][2])
        h = np.array([math.sqrt((RFID[pos][0]-x[0])*(RFID[pos][0]-x[0])+(RFID[pos][1]-x[1])*(RFID[pos][1]-x[1])),math.atan2(RFID[pos][1]-x[1],RFID[pos][0]-x[0])-x[2]])
        h += v_k
        H_ki = np.array([[(x[0]-RFID[pos][0])/h[0], (x[1]-RFID[pos][1])/h[0], 0],[-1*(x[1]-RFID[pos][1])/(h[0]*h[0]), (x[0]-RFID[pos][0])/(h[0]*h[0]), -1]])
        H_k.append(H_ki)
        h_k.append(h)
    H_k = np.concatenate(H_k,axis=0)
    h_k = np.concatenate(h_k,axis=0)

    z_k = z[:,:2]
    z_k = z_k.reshape(-1)

    y_t = z_k -  h_k# (8,1)
    S_k = np.matmul(np.matmul(H_k,P),np.transpose(H_k)) + R_k#(8,8)
    K_k = np.matmul(np.matmul(P,np.transpose(H_k)),np.linalg.inv(S_k))#(3,8)

    x_k = x + np.matmul(K_k,y_t)

    I = np.identity(3)
    P_k = np.matmul((I - np.matmul(K_k,H_k)),P)

    P = P_k
    x = x_k
    return x, P


def plot_covariance_ellipse(xEst, PEst):
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0]).flatten()
    py = np.array(fx[1, :] + xEst[1]).flatten()
    plt.plot(px, py, "--g")


def main():
    time = 0.0

    # Initialize state and covariance
    x_est = np.zeros(3)
    x_true = np.zeros(3)
    P = np.eye(3)

    # State history
    h_x_est = x_est.T
    h_x_true = x_true.T

    while time <= SIM_TIME:
        time += DT
        u = input(time, x_true)
        x_true = move(x_true, u)
        z = measure(x_true)
        x_est, P = EKF(x_est, P, u, z)

        # store data history
        h_x_est = np.vstack((h_x_est, x_est))
        h_x_true = np.vstack((h_x_true, x_true))

        if show_animation:
            plt.cla()

            for i in range(len(z[:,0])):
                plt.plot([x_true[0], RFID[int(z[i,2]),0]], [x_true[1], RFID[int(z[i,2]),1]], "-k")
            plt.plot(RFID[:,0], RFID[:,1], "*k")
            plt.plot(np.array(h_x_true[:,0]).flatten(),
                     np.array(h_x_true[:,1]).flatten(), "-b")
            plt.plot(np.array(h_x_est[:,0]).flatten(),
                     np.array(h_x_est[:,1]).flatten(), "-r")
            plot_covariance_ellipse(x_est, P)

            plt.axis("equal")
            plt.xlim(XLIM)
            plt.ylim(YLIM)
            plt.grid(True)
            plt.pause(0.001)

    plt.figure()
    errors = np.abs(h_x_true - h_x_est)
    plt.plot(errors)
    dth = errors[:,2] % (2*np.pi)
    errors[:,2] = np.amin(np.array([2*np.pi-dth, dth]), axis=0)
    plt.legend(['x error', 'y error', 'th error'])
    plt.xlabel('time')
    plt.ylabel('error magnitude')
    plt.ylim([0,1.5])
    plt.show()


if __name__ == '__main__':
    main()
