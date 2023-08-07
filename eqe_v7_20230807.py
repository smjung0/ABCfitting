import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.python import keras
import numpy as np
import matplotlib.pyplot as plt
import math

q = 1.602e-19
LEE = 0.25

# RED QD
j_train = [0.0001576,	0.001385,	0.00537,	0.01204,	0.02081,	0.03132,	0.04313,	0.05619,	0.0703,	0.0853,	0.1011,	0.1175,	0.1346,	0.1523,	0.1705, 0.1893,	0.2087,	0.2286,	0.2488,	0.2697]
e_train = [8.229,	9.8,	9.585,	9.329,	9.052,	8.847,	8.542,	8.323,	8.093,	7.894,	7.693,	7.512,	7.343,	7.188,	7.034, 6.907,	6.777,	6.648,	6.545,	6.433]
M = 1
dQD = 14.7e-7
LearningRate = 0.001

# GREEN QD
#j_train = [0.03478,	0.04903,	0.06489,	0.08189,	0.09991,	0.1187,	0.1381,	0.1581,	0.1785,	0.1994,	0.221,	0.2432]
#e_train = [6.827,	6.819,	6.762,	6.682,	6.592,	6.505,	6.424,	6.337,	6.26,	6.177,	6.102,	6.02]
#M = 2
#dQD = 8.7e-7
#LearningRate = 0.025

# BLUE QD
#j_train = [0.01275000, 0.02293000, 0.0367900, 0.0564500, 0.0777200, 0.1020000, 0.1239000, 0.1503000]
#e_train = [1.8000, 1.9860, 2.1440, 2.2320, 2.2780, 2.3050, 2.2880, 2.2770]
#M = 2
#dQD = 6.8e-7
#LearningRate = 0.4

emax = np.max(e_train)
imax = e_train.index(emax)
jmax = j_train[imax]
umax = jmax / (q * (M-0.5)*dQD)

def FJ(k, A, B, C):
    return q * (M-0.5)*dQD * (A * k + B * k * k + C * k * k * k)

def dFJ(k, A, B, C):
    return q * (M-0.5)*dQD * (A + 2. * B * k + 3. * C * k * k)

def FSolvek(A, B, C, Ji):
    k = 0
    for i in range(20):
        kp = k - (FJ(k, A, B, C) - Ji) / dFJ(k, A, B, C)
        k = kp
    return k

def FEQE(log10B, log10D):
    B = 10. ** log10B
    D = 10. ** log10D

    eta = emax / (LEE * D) / 100.
    Q = 2. * eta / (1 - eta)
    A = (B * umax / (Q * (2. + Q))) ** 0.5
    C = (B ** 2.) / (Q ** 2.) / A

    ndata = np.size(j_train)

    k = []

    for i in range(ndata):
        k.append(FSolvek(A, B, C, j_train[i]))

    np.transpose(k)
    eqe = 100. * (LEE * D * B * k / (A + B * k + C * k * k))
    return eqe


log10B = tf.Variable(tf.random.normal([1], -12.1135449, 0.001, tf.float64, seed=1), name='B')
log10D = tf.Variable(tf.random.normal([1], -0.001, 0.001, tf.float64, seed=1), name='D')

eqe = FEQE(log10B, log10D)

cost = tf.reduce_mean(tf.square(eqe - np.array(e_train)[:, np.newaxis]))  # tf.reduce_mean(tf.square(eqe - e_train)) # this line (for compatibility)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=LearningRate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(51):
    sess.run(train)
    rlgB = sess.run(log10B)
    rlgD = sess.run(log10D)
    rcost = sess.run(cost)

    B = 10. ** rlgB
    D = 10. ** rlgD

    eta = emax / (LEE * D) / 100.
    Q = 2. * eta / (1 - eta)
    A = (B * umax / (Q * (2. + Q))) ** 0.5
    C = (B ** 2.) / (Q ** 2.) / A
    eqe_opt = FEQE(rlgB, rlgD)
    print(step, rcost, A[0], B[0], C[0], D[0])
    #    print(sess.run(eqe)[6], e_train)
    if step % 5 == 0:
        plt.plot(j_train, eqe_opt, label="%d" % step)

plt.scatter(j_train, e_train, s=20, marker=f"o", alpha=1.0)

plt.legend(loc='lower right')

plt.show()