from math import *
import numpy as np

# Geometric parameters
l = 0.2828427
d = 0.4
# Dynamic parameters
mp = 3.0
mf = 1.0


def dgm(q11, q21, assembly_mode):
    OA1 = np.array([-d/2, q11])
    OA2 = np.array([d/2, q21])
    A2M = 0.5 * (OA1 - OA2)
    a = np.linalg.norm(A2M)
    h = np.sqrt(l**2 - a**2)
    MC = assembly_mode * (h/a) * (np.array([[0, -1], [1, 0]]) @ A2M)
    OC = OA2 + A2M + MC
    x = OC[0]
    y = OC[1]
    return x, y


def igm(x, y, gamma1, gamma2):
    q11 = y + gamma1 * np.sqrt(l**2 - (x + d/2)**2)
    q21 = y + gamma2 * np.sqrt(l**2 - (x - d/2)**2)
    return q11, q21


def dgm_passive(q11, q21, assembly_mode):
    x, y = dgm(q11, q21, assembly_mode)
    q12 = atan2(y - q11, x + d/2)
    q22 = atan2(y - q21, x - d/2)
    return q12, q22

def compute_unit_vectors(q12, q22):
    u1 = np.array([cos(q12), sin(q12)])
    u2 = np.array([cos(q22), sin(q22)])
    x0 = np.array([1, 0]) 
    y0 = np.array([0, 1])
    v1 = np.array([-u1[1], u1[0]]) #rotate u1 by 90 degrees ccw
    v2 = np.array([-u2[1], u2[0]]) #rotate u2 by 90 degrees ccw
    return u1, u2, x0, y0, v1, v2


# You can create intermediate functions to avoid redundant code
def compute_A_B(q11, q12, q21, q22):
    u1, u2, _, y0, _, _ = compute_unit_vectors(q12, q22)
    A = np.array([u1,u2])
    B = np.array([[np.dot(u1, y0), 0],[0, np.dot(u2, y0)]])
    return A, B


def dkm(q11, q12, q21, q22, q11D, q21D):
    A, B = compute_A_B(q11, q12, q21, q22)
    psiD = np.linalg.inv(A) @ B @ np.array([q11D, q21D])
    xD = psiD[0]
    yD = psiD[1]
    return xD, yD


def ikm(q11, q12, q21, q22, xD, yD):
    A, B = compute_A_B(q11, q12, q21, q22)
    qD = np.linalg.inv(B) @ A @ np.array([xD, yD])
    q11D = qD[0]
    q21D = qD[1]
    return q11D, q21D


def dkm_passive(q11, q12, q21, q22, q11D, q21D, xD, yD):
    _, _, _, y0, v1, v2 = compute_unit_vectors(q12, q22)
    A_passive = np.array([v1, v2])
    B_passive = np.array([[np.dot(v1, y0), 0], [0, np.dot(v2, y0)]])
    psiD = np.array([xD, yD])
    qD = np.array([q11D, q21D])
    qD_passive = (A_passive @ psiD - B_passive @ qD) / l
    q12D = qD_passive[0]
    q22D = qD_passive[1]
    return q12D, q22D
    

def dkm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD):
    A, B = compute_A_B(q11, q12, q21, q22)
    d = l*np.array([q12D**2, q22D**2])
    ksiDD = np.linalg.inv(A) @ (B @ np.array([q11DD, q21DD]) - d)
    xDD = ksiDD[0]
    yDD = ksiDD[1]
    return xDD, yDD


def ikm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, xDD, yDD):
    A, B = compute_A_B(q11, q12, q21, q22)
    d = l*np.array([q12D**2, q22D**2])
    qDD = np.linalg.inv(B) @ (A @ np.array([xDD, yDD]) + d)
    q11DD = qDD[0]
    q21DD = qDD[1]
    return q11DD, q21DD


def dkm2_passive(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD, xDD, yDD):
    _, _, _, y0, v1, v2 = compute_unit_vectors(q12, q22)
    A_passive = np.array([v1, v2])
    B_passive = np.array([[np.dot(v1, y0), 0], [0, np.dot(v2, y0)]])
    ksiDD = np.array([xDD, yDD])
    qDD = (A_passive @ ksiDD - B_passive @ np.array([q11DD, q21DD]))/l
    q12DD = qDD[0]
    q22DD = qDD[1]
    return q12DD, q22DD


def dynamic_model(q11, q12, q21, q22, q11D, q12D, q21D, q22D):
    A, B = compute_A_B(q11, q12, q21, q22)
    J = np.linalg.inv(A) @ B
    M = mf*np.eye(2) + J.T@J*mp
    c = -J.T @ np.linalg.inv(A) * mp * l @ np.array([q12D**2, q22D**2])
    return M, c
