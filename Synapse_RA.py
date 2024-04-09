import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import piece
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

L = 2 * np.pi;
sig2 = 0.5;
a=0;
np1 = pow(2, 7);
ddx = L/np1;
xx = np.linspace(-L / 2, L / 2 - ddx, np1);
GAUSS1 = 0.2 * 1 / (np.sqrt(2 * np.pi) * 0.8) * np.exp(-pow((xx - 0), 2) / pow(0.8, 2));
GAUSS2 = 0.2 * 1 / (np.sqrt(2 * np.pi) * 1) * np.exp(-pow((xx - 1.4), 2) / pow(1, 2));
I11 = GAUSS1 + GAUSS2# 0.4, 1.6

def relu1(x):
    if x > 0:
        a1 = x
    else:
        a1 = 0
    return a1

def relu2(x):
    if x < 0:
        a2 = -x
    else:
        a2 = 0
    return a2

def relu3(x):
    if x > 0:
        a3 = 0.4 * x
    else:
        a3 = 0
    return a3

def relu4(x):
    if x < 0:
        a4 = -0.4 * x
    else:
        a4 = 0
    return a4

# # ncells = 16
def topology_dis():
    X = np.zeros((128,128))
    x = np.linspace(0,np.pi,65)
    X[0, 0:65] = x
    x = sorted(np.linspace(0,np.pi,65),reverse=True)
    X[0, 65:128] = x[1:64]
    for i in range(1, 128):
        X[i,:] = np.roll(X[0,:], i)
    X[X < 2] = 0  #1.5
    X[X > 2] = -0.5
    return X

def piece(x):
    r=1/(1+np.exp(-20*(x-0.22)))
    return r

def gaussian_wave(x,H):
    return 1 * 1 / (np.sqrt(2 * np.pi) * 0.4) * np.exp(-(x+H) ** 2 / 0.4**2)

def dx1(x, t, I1, W):
    u = x[0: 128]
    q = x[128: 256]
    du = -u + 0.5 * piece(u) * q + np.dot(W,piece(u) * q) + I1
    dq = (1 - q) /alpha - 0.01 * beta * q * 1 / (1 + np.exp(-20 * (u - 0.22)));
    return np.concatenate([du, dq])

def dx2(x, t, H, v, c):
    a1 = piece.relu1(v);
    a2 = piece.relu2(v);
    a3 = piece.relu3(v);
    a4 = piece.relu4(v);
    x2 = xx - c * t
    x3 = (x2 + np.pi) % (2 * np.pi) - np.pi
    Z = gaussian_wave(x3,H)
    u0, u1, u2, u3, u4, u5, u6, u7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    u8, u9, u10, u11, u12, u13, u14, u15 = x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]
    q0, q1, q2, q3, q4, q5, q6, q7 = x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23]
    q8, q9, q10, q11, q12, q13, q14, q15 = x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31]
    du0 = -u0 + a0 * piece.piece2(u0) * q0 + a1 * piece.piece2(u15) * q15 + a3 * piece.piece2(
        u14) * q14 + a2 * piece.piece2(u1) * q1 + a4 * piece.piece2(u2) * q2 + a5 * piece.piece2(
        u5) * q5 + a6 * piece.piece2(u6) * q6 + a7 * piece.piece2(u7) * q7 + a7 * piece.piece2(
        u8) * q8 + a7 * piece.piece2(u9) * q9 + a7 * piece.piece2(u10) * q10 + a7 * piece.piece2(u11) * q11 + Z[0];
    du1 = -u1 + a0 * piece.piece2(u1) * q1 + a1 * piece.piece2(u0) * q0 + a3 * piece.piece2(
        u15) * q15 + a2 * piece.piece2(u2) * q2 + a4 * piece.piece2(u3) * q3 + a5 * piece.piece2(
        u6) * q6 + a6 * piece.piece2(u7) * q7 + a7 * piece.piece2(u8) * q8 + a7 * piece.piece2(
        u9) * q9 + a7 * piece.piece2(u10) * q10 + a7 * piece.piece2(u11) * q11 + a7 * piece.piece2(u12) * q12 + Z[1];
    du2 = -u2 + a0 * piece.piece2(u2) * q2 + a1 * piece.piece2(u1) * q1 + a3 * piece.piece2(
        u0) * q0 + a2 * piece.piece2(u3) * q3 + a4 * piece.piece2(u4) * q4+ a5 * piece.piece2(
        u7) * q7 + a6 * piece.piece2(u8) * q8 + a7 * piece.piece2(u9) * q9 + a7 * piece.piece2(
        u10) * q10 + a7 * piece.piece2(u11) * q11 + a7 * piece.piece2(u12) * q12 + a7 * piece.piece2(u13) * q13 + Z[2];
    du3 = -u3 + a0 * piece.piece2(u3) * q3 + a1 * piece.piece2(u2) * q2 + a3 * piece.piece2(
        u1) * q1 + a2 * piece.piece2(u4) * q4 + a4 * piece.piece2(u5) * q5+ a5 * piece.piece2(
        u8) * q8 + a6 * piece.piece2(u9) * q9 + a7 * piece.piece2(u10) * q10 + a7 * piece.piece2(
        u11) * q11 + a7 * piece.piece2(u12) * q12 + a7 * piece.piece2(u13) * q13 + a7 * piece.piece2(u14) * q14 + Z[3];
    du4 = -u4 + a0 * piece.piece2(u4) * q4 + a1 * piece.piece2(u3) * q3 + a3 * piece.piece2(
        u2) * q2 + a2 * piece.piece2(u5) * q5 + a4 * piece.piece2(u6) * q6 + a5 * piece.piece2(
        u9) * q9 + a6 * piece.piece2(u10) * q10 + a7 * piece.piece2(u11) * q11 + a7 * piece.piece2(
        u12) * q12 + a7 * piece.piece2(u13) * q13 + a7 * piece.piece2(u14) * q14 + a7 * piece.piece2(u15) * q15 + Z[4];
    du5 = -u5 + a0 * piece.piece2(u5) * q5 + a1 * piece.piece2(u4) * q4 + a3 * piece.piece2(
        u3) * q3 + a2 * piece.piece2(u6) * q6 + a4 * piece.piece2(u7) * q7 + a5 * piece.piece2(
        u10) * q10 + a6 * piece.piece2(u11) * q11 + a7 * piece.piece2(u12) * q12 + a7 * piece.piece2(
        u13) * q13 + a7 * piece.piece2(u14) * q14 + a7 * piece.piece2(u15) * q15 + a7 * piece.piece2(u0) * q0 + Z[5];
    du6 = -u6 + a0 * piece.piece2(u6) * q6 + a1 * piece.piece2(u5) * q5 + a3 * piece.piece2(
        u4) * q4 + a2 * piece.piece2(u7) * q7 + a4 * piece.piece2(u8) * q8 + a5 * piece.piece2(
        u11) * q11 + a6 * piece.piece2(u12) * q12 + a7 * piece.piece2(u13) * q13 + a7 * piece.piece2(
        u14) * q14 + a7 * piece.piece2(u15) * q15 + a7 * piece.piece2(u0) * q0 + a7 * piece.piece2(u1) * q1 + Z[6];
    du7 = -u7 + a0 * piece.piece2(u7) * q7 + a1 * piece.piece2(u6) * q6 + a3 * piece.piece2(
        u5) * q5 + a2 * piece.piece2(u8) * q8 + a4 * piece.piece2(u9) * q9 + a5 * piece.piece2(
        u12) * q12 + a6 * piece.piece2(u13) * q13 + a7 * piece.piece2(u14) * q14 + a7 * piece.piece2(
        u15) * q15 + a7 * piece.piece2(u0) * q0 + a7 * piece.piece2(u1) * q1 + a7 * piece.piece2(u2) * q2 + Z[7];
    du8 = -u8 + a0 * piece.piece2(u8) * q8 + a1 * piece.piece2(u7) * q7 + a3 * piece.piece2(
        u6) * q6 + a2 * piece.piece2(u9) * q9 + a4 * piece.piece2(u10) * q10 + a5 * piece.piece2(
        u13) * q13 + a6 * piece.piece2(u14) * q14 + a7 * piece.piece2(u15) * q15 + a7 * piece.piece2(
        u0) * q0 + a7 * piece.piece2(u1) * q1 + a7 * piece.piece2(u2) * q2 + a7 * piece.piece2(u3) * q3 + Z[8];
    du9 = -u9 + a0 * piece.piece2(u9) * q9 + a1 * piece.piece2(u8) * q8 + a3 * piece.piece2(
        u7) * q7 + a2 * piece.piece2(u10) * q10 + a4 * piece.piece2(u11) * q11 + a5 * piece.piece2(
        u14) * q14 + a6 * piece.piece2(u15) * q15 + a7 * piece.piece2(u0) * q0 + a7 * piece.piece2(
        u1) * q1 + a7 * piece.piece2(u2) * q2 + a7 * piece.piece2(u3) * q3 + a7 * piece.piece2(u4) * q4 + Z[9];
    du10 = -u10 + a0 * piece.piece2(u10) * q10 + a1 * piece.piece2(u9) * q9 + a3 * piece.piece2(
        u8) * q8 + a2 * piece.piece2(u11) * q11 + a4 * piece.piece2(u12) * q12 + a5 * piece.piece2(
        u15) * q15 + a6 * piece.piece2(u0) * q0 + a7 * piece.piece2(u1) * q1 + a7 * piece.piece2(
        u2) * q2 + a7 * piece.piece2(u3) * q3 + a7 * piece.piece2(u4) * q4 + a7 * piece.piece2(u5) * q5 + Z[10];
    du11 = -u11 + a0 * piece.piece2(u11) * q11 + a1 * piece.piece2(u10) * q10 + a3 * piece.piece2(
        u9) * q9 + a2 * piece.piece2(u12) * q12 + a4 * piece.piece2(u13) * q13 + a5 * piece.piece2(
        u0) * q0 + a6 * piece.piece2(u1) * q1 + a7 * piece.piece2(u2) * q2 + a7 * piece.piece2(
        u3) * q3 + a7 * piece.piece2(u4) * q4 + a7 * piece.piece2(u5) * q5 + a7 * piece.piece2(u6) * q6 + Z[11];
    du12 = -u12 + a0 * piece.piece2(u12) * q12 + a1 * piece.piece2(u11) * q11 + a3 * piece.piece2(
        u10) * q10 + a2 * piece.piece2(u13) * q13 + a4 * piece.piece2(u14) * q14 + a5 * piece.piece2(
        u1) * q1 + a6 * piece.piece2(u2) * q2 + a7 * piece.piece2(u3) * q3 + a7 * piece.piece2(
        u4) * q4 + a7 * piece.piece2(u5) * q5 + a7 * piece.piece2(u6) * q6 + a7 * piece.piece2(u7) * q7 + Z[12];
    du13 = -u13 + a0 * piece.piece2(u13) * q13 + a1 * piece.piece2(u12) * q12 + a3 * piece.piece2(
        u11) * q11 + a2 * piece.piece2(u14) * q14 + a4 * piece.piece2(u15) * q15 + a5 * piece.piece2(
        u2) * q2 + a6 * piece.piece2(u3) * q3 + a7 * piece.piece2(u4) * q4 + a7 * piece.piece2(
        u5) * q5 + a7 * piece.piece2(u6) * q6 + a7 * piece.piece2(u7) * q7 + a7 * piece.piece2(u8) * q8 + Z[13];
    du14 = -u14 + a0 * piece.piece2(u14) * q14 + a1 * piece.piece2(u13) * q13 + a3 * piece.piece2(
        u12) * q12 + a2 * piece.piece2(u15) * q15 + a4 * piece.piece2(u0) * q0 + a5 * piece.piece2(
        u3) * q3 + a6 * piece.piece2(u4) * q4 + a7 * piece.piece2(u5) * q5 + a7 * piece.piece2(
        u6) * q6 + a7 * piece.piece2(u7) * q7 + a7 * piece.piece2(u8) * q8 + a7 * piece.piece2(u9) * q9 + Z[14];
    du15 = -u15 + a0 * piece.piece2(u15) * q15 + a1 * piece.piece2(u14) * q14 + a3 * piece.piece2(
        u13) * q13 + a2 * piece.piece2(u0) * q0 + a4 * piece.piece2(u1) * q1 + a5 * piece.piece2(
        u4) * q4 + a6 * piece.piece2(u5) * q5 + a7 * piece.piece2(u6) * q6 + a7 * piece.piece2(
        u7) * q7 + a7 * piece.piece2(u8) * q8 + a7 * piece.piece2(u9) * q9 + a7 * piece.piece2(u10) * q10 + Z[15];
    dq0 = (1 - q0) /alpha - 0.01 * beta * q0 * 1 / (1 + np.exp(-20 * (u0 - 0.22)));
    dq1 = (1 - q1) /alpha - 0.01 * beta * q1 * 1 / (1 + np.exp(-20 * (u1 - 0.22)));
    dq2 = (1 - q2) /alpha - 0.01 * beta * q2 * 1 / (1 + np.exp(-20 * (u2 - 0.22)));
    dq3 = (1 - q3) /alpha - 0.01 * beta * q3 * 1 / (1 + np.exp(-20 * (u3 - 0.22)));
    dq4 = (1 - q4) /alpha - 0.01 * beta * q4 * 1 / (1 + np.exp(-20 * (u4 - 0.22)));
    dq5 = (1 - q5) /alpha - 0.01 * beta * q5 * 1 / (1 + np.exp(-20 * (u5 - 0.22)));
    dq6 = (1 - q6) /alpha - 0.01 * beta * q6 * 1 / (1 + np.exp(-20 * (u6 - 0.22)));
    dq7 = (1 - q7) /alpha - 0.01 * beta * q7 * 1 / (1 + np.exp(-20 * (u7 - 0.22)));
    dq8 = (1 - q8) / alpha - 0.01 * beta * q8 * 1 / (1 + np.exp(-20 * (u8 - 0.22)));
    dq9 = (1 - q9) / alpha - 0.01 * beta * q9 * 1 / (1 + np.exp(-20 * (u9 - 0.22)));
    dq10 = (1 - q10) / alpha - 0.01 * beta * q10 * 1 / (1 + np.exp(-20 * (u10 - 0.22)));
    dq11 = (1 - q11) / alpha - 0.01 * beta * q11 * 1 / (1 + np.exp(-20 * (u11 - 0.22)));
    dq12 = (1 - q12) / alpha - 0.01 * beta * q12 * 1 / (1 + np.exp(-20 * (u12 - 0.22)));
    dq13 = (1 - q13) / alpha - 0.01 * beta * q13 * 1 / (1 + np.exp(-20 * (u13 - 0.22)));
    dq14 = (1 - q14) / alpha - 0.01 * beta * q14 * 1 / (1 + np.exp(-20 * (u14 - 0.22)));
    dq15 = (1 - q15) / alpha - 0.01 * beta * q15 * 1 / (1 + np.exp(-20 * (u15 - 0.22)));
    return [du0,du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,du12,du13,du14,du15,dq0,dq1,dq2,dq3,dq4,dq5,dq6,dq7,dq8,dq9,dq10,dq11,dq12,dq13,dq14,dq15]


m1 = np.zeros((5000, 8));
beta = 1;
alpha = 400;
v = 0.;
a1 = relu1(v); a2 = relu2(v); a3 = relu3(v); a4 = relu4(v);
a0 = 0.5; a5 = -0.5; a6 = -0.5; a7 = -0.5;
x0 = np.zeros(256);

for i in range(0, len(x0)):
    t = np.linspace(0,1000,1001)
    W = topology_dis()
    xd = odeint(dx1, x0, t, args=(I11,W,))


print()