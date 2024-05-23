import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

L = 2 * np.pi;
np1 = pow(2, 4);
ddx = L/np1;
xx = np.linspace(-L / 2, L / 2 - ddx, np1);


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

def piece(x):
    r=1/(1+np.exp(-20*(x-0.22)))
    return r

def gaussian_wave(x,H):
    return 1 * 1 / (np.sqrt(2 * np.pi) * 0.4) * np.exp(-(x + H) ** 2 / 0.4**2)

def dx2(x, t, H, v, c):
    a1 = relu1(v);
    a2 = relu2(v);
    a3 = relu3(v);
    a4 = relu4(v);
    x2 = xx - c * t
    x3 = (x2 + np.pi) % (2 * np.pi) - np.pi
    Z = gaussian_wave(x3,H)
    u0, u1, u2, u3, u4, u5, u6, u7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    u8, u9, u10, u11, u12, u13, u14, u15 = x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]
    q0, q1, q2, q3, q4, q5, q6, q7 = x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23]
    q8, q9, q10, q11, q12, q13, q14, q15 = x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31]
    du0 = -u0 + a0 * piece(u0) * q0 + a1 * piece(u15) * q15 + a3 * piece(u14) * q14 + a2 * piece(u1) * q1 + a4 * piece(
        u2) * q2 + a5 * piece(u5) * q5 + a6 * piece(u6) * q6 + a7 * piece(u7) * q7 + a7 * piece(u8) * q8 + a7 * piece(
        u9) * q9 + a7 * piece(u10) * q10 + a7 * piece(u11) * q11 + Z[0];
    du1 = -u1 + a0 * piece(u1) * q1 + a1 * piece(u0) * q0 + a3 * piece(u15) * q15 + a2 * piece(u2) * q2 + a4 * piece(
        u3) * q3 + a5 * piece(u6) * q6 + a6 * piece(u7) * q7 + a7 * piece(u8) * q8 + a7 * piece(u9) * q9 + a7 * piece(
        u10) * q10 + a7 * piece(u11) * q11 + a7 * piece(u12) * q12 + Z[1];
    du2 = -u2 + a0 * piece(u2) * q2 + a1 * piece(u1) * q1 + a3 * piece(u0) * q0 + a2 * piece(u3) * q3 + a4 * piece(
        u4) * q4+ a5 * piece(u7) * q7 + a6 * piece(u8) * q8 + a7 * piece(u9) * q9 + a7 * piece(u10) * q10 + a7 * piece(
        u11) * q11 + a7 * piece(u12) * q12 + a7 * piece(u13) * q13 + Z[2];
    du3 = -u3 + a0 * piece(u3) * q3 + a1 * piece(u2) * q2 + a3 * piece(u1) * q1 + a2 * piece(u4) * q4 + a4 * piece(
        u5) * q5+ a5 * piece(u8) * q8 + a6 * piece(u9) * q9 + a7 * piece(u10) * q10 + a7 * piece(u11) * q11 + a7 * piece(
        u12) * q12 + a7 * piece(u13) * q13 + a7 * piece(u14) * q14 + Z[3];
    du4 = -u4 + a0 * piece(u4) * q4 + a1 * piece(u3) * q3 + a3 * piece(u2) * q2 + a2 * piece(u5) * q5 + a4 * piece(
        u6) * q6 + a5 * piece(u9) * q9 + a6 * piece(u10) * q10 + a7 * piece(u11) * q11 + a7 * piece(
        u12) * q12 + a7 * piece(u13) * q13 + a7 * piece(u14) * q14 + a7 * piece(u15) * q15 + Z[4];
    du5 = -u5 + a0 * piece(u5) * q5 + a1 * piece(u4) * q4 + a3 * piece(u3) * q3 + a2 * piece(u6) * q6 + a4 * piece(
        u7) * q7 + a5 * piece(u10) * q10 + a6 * piece(u11) * q11 + a7 * piece(u12) * q12 + a7 * piece(
        u13) * q13 + a7 * piece(u14) * q14 + a7 * piece(u15) * q15 + a7 * piece(u0) * q0 + Z[5];
    du6 = -u6 + a0 * piece(u6) * q6 + a1 * piece(u5) * q5 + a3 * piece(u4) * q4 + a2 * piece(u7) * q7 + a4 * piece(
        u8) * q8 + a5 * piece(u11) * q11 + a6 * piece(u12) * q12 + a7 * piece(u13) * q13 + a7 * piece(
        u14) * q14 + a7 * piece(u15) * q15 + a7 * piece(u0) * q0 + a7 * piece(u1) * q1 + Z[6];
    du7 = -u7 + a0 * piece(u7) * q7 + a1 * piece(u6) * q6 + a3 * piece(u5) * q5 + a2 * piece(u8) * q8 + a4 * piece(
        u9) * q9 + a5 * piece(u12) * q12 + a6 * piece(u13) * q13 + a7 * piece(u14) * q14 + a7 * piece(
        u15) * q15 + a7 * piece(u0) * q0 + a7 * piece(u1) * q1 + a7 * piece(u2) * q2 + Z[7];
    du8 = -u8 + a0 * piece(u8) * q8 + a1 * piece(u7) * q7 + a3 * piece(u6) * q6 + a2 * piece(u9) * q9 + a4 * piece(
        u10) * q10 + a5 * piece(u13) * q13 + a6 * piece(u14) * q14 + a7 * piece(u15) * q15 + a7 * piece(
        u0) * q0 + a7 * piece(u1) * q1 + a7 * piece(u2) * q2 + a7 * piece(u3) * q3 + Z[8];
    du9 = -u9 + a0 * piece(u9) * q9 + a1 * piece(u8) * q8 + a3 * piece(u7) * q7 + a2 * piece(u10) * q10 + a4 * piece(
        u11) * q11 + a5 * piece(u14) * q14 + a6 * piece(u15) * q15 + a7 * piece(u0) * q0 + a7 * piece(
        u1) * q1 + a7 * piece(u2) * q2 + a7 * piece(u3) * q3 + a7 * piece(u4) * q4 + Z[9];
    du10 = -u10 + a0 * piece(u10) * q10 + a1 * piece(u9) * q9 + a3 * piece(u8) * q8 + a2 * piece(u11) * q11 + a4 * piece(
        u12) * q12 + a5 * piece(u15) * q15 + a6 * piece(u0) * q0 + a7 * piece(u1) * q1 + a7 * piece(
        u2) * q2 + a7 * piece(u3) * q3 + a7 * piece(u4) * q4 + a7 * piece(u5) * q5 + Z[10];
    du11 = -u11 + a0 * piece(u11) * q11 + a1 * piece(u10) * q10 + a3 * piece(u9) * q9 + a2 * piece(u12) * q12 + a4 * piece(
        u13) * q13 + a5 * piece(u0) * q0 + a6 * piece(u1) * q1 + a7 * piece(u2) * q2 + a7 * piece(u3) * q3 + a7 * piece(
        u4) * q4 + a7 * piece(u5) * q5 + a7 * piece(u6) * q6 + Z[11];
    du12 = -u12 + a0 * piece(u12) * q12 + a1 * piece(u11) * q11 + a3 * piece(u10) * q10 + a2 * piece(u13) * q13 + a4 * piece(
        u14) * q14 + a5 * piece(u1) * q1 + a6 * piece(u2) * q2 + a7 * piece(u3) * q3 + a7 * piece(u4) * q4 + a7 * piece(
        u5) * q5 + a7 * piece(u6) * q6 + a7 * piece(u7) * q7 + Z[12];
    du13 = -u13 + a0 * piece(u13) * q13 + a1 * piece(u12) * q12 + a3 * piece(u11) * q11 + a2 * piece(u14) * q14 + a4 * piece(
        u15) * q15 + a5 * piece(u2) * q2 + a6 * piece(u3) * q3 + a7 * piece(u4) * q4 + a7 * piece(u5) * q5 + a7 * piece(
        u6) * q6 + a7 * piece(u7) * q7 + a7 * piece(u8) * q8 + Z[13];
    du14 = -u14 + a0 * piece(u14) * q14 + a1 * piece(u13) * q13 + a3 * piece(u12) * q12 + a2 * piece(u15) * q15 + a4 * piece(
        u0) * q0 + a5 * piece(u3) * q3 + a6 * piece(u4) * q4 + a7 * piece(u5) * q5 + a7 * piece(u6) * q6 + a7 * piece(
        u7) * q7 + a7 * piece(u8) * q8 + a7 * piece(u9) * q9 + Z[14];
    du15 = -u15 + a0 * piece(u15) * q15 + a1 * piece(u14) * q14 + a3 * piece(u13) * q13 + a2 * piece(u0) * q0 + a4 * piece(
        u1) * q1 + a5 * piece(u4) * q4 + a6 * piece(u5) * q5 + a7 * piece(u6) * q6 + a7 * piece(u7) * q7 + a7 * piece(
        u8) * q8 + a7 * piece(u9) * q9 + a7 * piece(u10) * q10 + Z[15];
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
alpha = 40;
a0, a5, a6, a7= 0.5, -0.5, -0.5,- 0.5;
# initial value
x0 = np.array([-1.86,-1.43,-0.98,-0.52,0.33,0.79,0.97,0.98,0.98,0.20,-0.63,-1.27,-1.75,-2.2,-2.28,-2.27,0.83,0.83,0.84,0.84,0.84,0.87,0.89,0.89,0.9,0.93,0.91,0.89,0.89,0.87,0.86,0.84]);
t = np.linspace(0,200,201)
xd = odeint(dx2, x0, t, args=(0, 0.43, 0.09))


fig = plt.figure(figsize=(6, 6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
y_ticks = np.round(xx, 1);  # 自定义横纵轴标签
fig = sns.heatmap(xd.T[0: 16, 0:100],annot=False,fmt='.3f',cmap="Reds",vmax=2,vmin=0,xticklabels=10, yticklabels=y_ticks)
cbar = fig.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
plt.xticks([0,20,40,60,80],['0','200','400','600','800'],fontsize=18)
plt.yticks([0,4,8,12,16],['-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'],fontsize=18)
plt.vlines(400, -200, 200, colors='k', linestyles='dashed',linewidth=3)
fig.invert_yaxis()
fig.set_xlabel('t (ms)', fontsize=18);
fig.set_ylabel('Neurons Label x\n(radian)', fontsize=18);
plt.show()
print()