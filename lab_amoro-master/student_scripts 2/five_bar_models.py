from math import *
import numpy as np

# Geometric parameters
l = 0.09
d = 0.118
# Dynamic parameters
ZZ1R = 1.0 * 0.045 * 0.045
ZZ2R = 1.0 * 0.045 * 0.045
mR = 0.5


def dgm(q11, q21, assembly_mode):
	A22H = 0.5 * np.array([[- l * np.cos(q21) - d + l * np.cos(q11)], [-l * np.sin(q21) + l * np.sin(q11)]])
	OA21 = np.array([[0.5 * d], [0.0]])
	A21A22 = np.array([[l * np.cos(q21)], [l * np.sin(q21)]])
	a = 0.5 * d + l * np.cos(q21)
	#a = np.linalg.norm(A22H)
	h = np.sqrt(l**2 - a**2)
	#h = sqrt(abs(l**2 - a**2))
	HA13 = assembly_mode * (h/a) * (np.array([[0.0, -1.0], [1.0, 0.0]]) @ A22H)
	A22H = np.array([[-a], [0.0]])
	OA13 = OA21 + A21A22 + A22H + HA13
	x = OA13[0]
	y = OA13[1] 
	return x, y


def igm(x, y, gamma1, gamma2):
	A11A13 = np.array([x + 0.5 * d, y])
	A11M1 = 0.5 * A11A13
	c = np.linalg.norm(A11M1)
	b = np.sqrt(l**2 - c**2)
	M1A12 = gamma1 * (b/c) * (np.array([[0, -1], [1.0, 0.0]]) @ A11M1)
	A11A12 = A11M1 + M1A12  
	q11 = np.arctan2(A11A12[1], A11A12[0])
	
	A21A13 = np.array([ x - d * 0.5, y])
	A21M2 = 0.5 * A21A13
	c = np.linalg.norm(A21M2)
	b = np.sqrt(l**2 - c**2)
	M2A22 = gamma2 * (b/c) * (np.array([[0, -1], [1.0, 0.0]]) @ A21M2)
	A21A22 = A21M2 + M2A22
	q21 = np.arctan2(A21A22[1], A21A22[0])
	return q11, q21


def dgm_passive(q11, q21, assembly_mode):
	x,y = dgm(q11, q21, assembly_mode)
	q12 = np.arctan2(y/l - np.sin(q11), x/l + d/(2*l) - np.cos(q11)) - q11
	q22 = np.arctan2(y/l - np.sin(q21), x/l + d/(2*l) - np.cos(q21)) - q21
	return q12, q22


# You can create intermediate functions to avoid redundant code
def compute_u11(q11):
	u11 = np.array([cos(q11), sin(q11)])
	return u11

def compute_u12(q11, q12):
	u12 = np.array([cos(q11 + q12), sin(q11 + q12)])
	return u12
	
def compute_u21(q21):
	u21 = np.array([cos(q21), sin(q21)])
	return u21
	
def compute_u22(q21, q22):
	u22 = np.array([cos(q21 + q22), sin(q21 + q22)])
	return u22
	
def rotate_90_counterclockwise(v):
	rotation_matrix = np.array([[0, -1], [1,  0]])
	return rotation_matrix @ v
	
def compute_v11(u11):
	v11 = rotate_90_counterclockwise(u11)
	return v11

def compute_v12(u12):
	v12 = rotate_90_counterclockwise(u12)
	return v12
	
def compute_v21(u21):
	v21 = rotate_90_counterclockwise(u21)
	return v21
		
def compute_v22(u22):
	v22 = rotate_90_counterclockwise(u22)
	return v22

def compute_A_B(q11, q12, q21, q22):
	u12 = compute_u12(q11, q12)
	u22 = compute_u22(q21, q22)
	v11 = compute_v11(compute_u11(q11))
	v21 = compute_v21(compute_u21(q21))
	A = np.array([u12, u22])
	B = np.array([[l * u12.dot(v11), 0], [0, l * u22.dot(v21)]])
	return A, B


def dkm(q11, q12, q21, q22, q11D, q21D):
	A,B = compute_A_B(q11, q12, q21, q22)
	qD = np.array([q11D, q21D])
	psiD = np.linalg.inv(A) @ B @ qD
	xD = psiD[0]
	yD = psiD[1]
	return xD, yD


def ikm(q11, q12, q21, q22, xD, yD):
	A,B = compute_A_B(q11, q12, q21, q22)
	psiD = np.array([xD, yD])
	qD =  np.linalg.inv(B) @ A @ psiD 
	q11D = qD[0]
	q21D = qD[1]
	return q11D, q21D


def dkm_passive(q11, q12, q21, q22, q11D, q21D, xD, yD):
	v12 = compute_v12(compute_u12(q11, q12))
	v22 = compute_v22(compute_u22(q21, q22))
	v11 = compute_v11(compute_u11(q11))
	v21 = compute_v21(compute_u21(q21))
	v = np.array([v12, v22])
	psiD = np.array([xD, yD])
	
	a = l*v12.dot(v11) + l
	b = l*v22.dot(v21) + l
	C = np.array([[a,0], [0,b]])
	qD = np.array([q11D, q21D])
	qpD = (1/l) * (v.dot(psiD) - C @ qD)
	q12D = qpD[0]
	q22D = qpD[1]
	return q12D, q22D
	
	
def compute_d_matrix(q11, q21, q12, q22, q11D, q12D, q21D, q22D, l):
	u11 = compute_u11(q11)
	u21 = compute_u21(q21)
	u12 = compute_u12(q11, q12)
	u22 = compute_u22(q21, q22)
	return np.array([-l * q11D**2 * np.dot(u12.T, u11) - l*(q11D + q12D)**2, -l * q21D**2 * np.dot(u22.T, u21) - l*(q21D + q22D)**2])


def dkm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD):
    A, B = compute_A_B(q11, q12, q21, q22)
    d_matrix = compute_d_matrix(q11, q21, q12, q22, q11D, q12D, q21D, q22D, l)
    qDD = np.array([q11DD, q21DD])
    ksiDD = np.linalg.inv(A) @ (B @ qDD + d_matrix) 
    xDD = ksiDD[0]
    yDD = ksiDD[1]
    return xDD, yDD


def ikm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, xDD, yDD):
    A, B = compute_A_B(q11, q12, q21, q22)
    d_matrix = compute_d_matrix(q11, q21, q12, q22, q11D, q12D, q21D, q22D, l)
    ksiDD = np.array([xDD, yDD])
    qDD = np.linalg.inv(B) @ ((A @ ksiDD) - d_matrix) 
    q11DD = qDD[0]
    q21DD = qDD[1]
    return q11DD, q21DD


def dkm2_passive(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD, xDD, yDD):
    v12 = compute_v12(compute_u12(q11, q12))
    v22 = compute_v22(compute_u22(q21, q22))
    v11 = compute_v11(compute_u11(q11))
    v21 = compute_v21(compute_u21(q21))
    u11 = compute_u11(q11)
    u21 = compute_u21(q21)
    v = np.array([v12, v22])
    psiDD = np.array([xDD, yDD])
    B = np.array([[l*v12.dot(v11) + l,0], [0,l*v22.dot(v21) + l]])
    d_passive = np.array([-l * q11D**2 *np.dot(v12, u11), -l * q21D**2 * np.dot(v22, u21)])
    qDD = np.array([q11DD, q21DD])
    qpDD = (1/l) * (v.dot(psiDD) - d_passive - B @ qDD)
    q12DD = qpDD[0]
    q22DD = qpDD[1]
    return q12DD, q22DD


def dynamic_model(q11, q12, q21, q22, q11D, q12D, q21D, q22D):
	Z = np.array([[ZZ1R, 0], [0, ZZ2R]])
	A, B = compute_A_B(q11, q12, q21, q22)
	d_matrix = compute_d_matrix(q11, q21, q12, q22, q11D, q12D, q21D, q22D, l)
	J = np.dot(np.linalg.inv(A), B)
	M = Z + mR*J.T @ np.linalg.inv(A) @ B
	c = mR * J.T @ np.linalg.inv(A) @ d_matrix
	return M, c
