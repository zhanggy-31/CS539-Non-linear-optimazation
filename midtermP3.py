from numpy import array
import numpy
import matplotlib.pyplot as plt
import math

v1, v2, theta2 = 1., 1., 0.
v1_true, v2_true, theta2_true = 1.05, 0.98, 0.1
g, b = 1., -5.


def h1(x):
    h1 = g * (x[0] * x[1] * math.cos(x[2]) - pow(x[0], 2)) - b * x[0] * x[1] * math.sin(x[2])
    return h1


def h2(x):
    h2 = (-b) * (x[0] * x[1] * math.cos(x[2]) - pow(x[0], 2)) - g * x[0] * x[1] * math.sin(x[2])
    return h2


def h3(x):
    h3 = g * (x[1] * x[0] * math.cos(x[2]) - pow(x[1], 2)) + b * x[1] * x[0] * math.sin(x[2])
    return h3


def h4(x):
    h4 = (-b) * (x[1] * x[0] * math.cos(x[2]) - pow(x[1], 2)) + g * x[1] * x[0] * math.sin(x[2])
    return h4


def dfxdv_1(x, p12, p21, q12, q21):
    dfxdv1 = -2 * (p12 - h1(x)) * (g * (x[1] * math.cos(x[2]) - 2 * x[0]) - b * x[1] * math.sin(x[2]))\
             -2 * (q12 - h2(x)) * (-b * (x[1] * math.cos(x[2]) - 2 * x[0]) - g * x[1] * math.sin(x[2]))\
             -2 * (p21 - h3(x)) * (g * x[1] * math.cos(x[2]) + b * x[1] * math.sin(x[2]))\
             -2 * (q21 - h4(x)) * (-b * x[1] * math.cos(x[2]) + g * x[1] * math.sin(x[2]))
    return dfxdv1


def dfxdv_2(x, p12, p21, q12, q21):
    dfxdv2 = -2 * (p12 - h1(x)) * (g * x[0] * math.cos(x[2]) - b * x[0] * math.sin(x[2]))\
             -2 * (q12 - h2(x)) * (-b * x[0] * math.cos(x[2]) - g * x[0] * math.sin(x[2]))\
             -2 * (p21 - h3(x)) * (g * (x[0] * math.cos(x[2]) - 2 * x[1]) + b * x[0] * math.sin(x[2]))\
             -2 * (q21 - h4(x)) * (-b * (x[0] * math.cos(x[2]) - 2 * x[1]) - g * x[0] * math.sin(x[2]))
    return dfxdv2


def dfxdt_2(x, p12, p21, q12, q21):
    dfxdt2 = -2 * (p12 - h1(x)) * (-g * x[0] * x[1] * math.sin(x[2]) - b * x[0] * x[1] * math.cos(x[2]))\
             -2 * (q12 - h2(x)) * (b * x[0] * x[1] * math.sin(x[2]) - g * x[0] * x[1] * math.cos(x[2]))\
             -2 * (p21 - h3(x)) * (-g * x[1] * x[0] * math.sin(x[2]) + b * x[1] * x[0] * math.cos(x[2]))\
             -2 * (q21 - h4(x)) * (b * x[1] * x[0] * math.sin(x[2]) + g * x[1] * x[0] * math.cos(x[2]))
    return dfxdt2


def fx(x, p12, p21, q12, q21):
    fx = pow((p12 - h1(x)), 2) + pow((q12 - h2(x)), 2) + pow((p21 - h3(x)), 2) + pow((q21 - h4(x)), 2)

    return fx


def get_alpha(x, alpha, eta, gx, p12, p21, q12, q21, e):
    while fx(x + alpha * -gx, p12, p21, q12, q21) > fx(x, p12, p21, q12, q21) + e * alpha * gx.dot(-gx):
        alpha = alpha / eta
    return alpha


def solve():
    x0 = [v1, v2, theta2]
    x_true = [v1_true, v2_true, theta2_true]
    xk = x0
    alpha = 1.0
    epsilon = 1e-1
    e = 0.5
    eta = 2
    k = 0
    g_result = []
    x_result = []
    p12 = h1(x_true)
    q12 = h2(x_true)
    p21 = h3(x_true)
    q21 = h4(x_true)
    dv1 = dfxdv_1(x0, p12, p21, q12, q21)
    dv2 = dfxdv_2(x0, p12, p21, q12, q21)
    dt2 = dfxdt_2(x0, p12, p21, q12, q21)
    gk = array([dv1, dv2, dt2])
    while numpy.linalg.norm(gk) > epsilon:
        alpha = get_alpha(xk, alpha, eta, gk, p12, p21, q12, q21, e)
        xk = xk - alpha * gk
        x_result.append(pow(numpy.linalg.norm(xk-x_true), 2))
        dv1 = dfxdv_1(xk, p12, p21, q12, q21)
        dv2 = dfxdv_2(xk, p12, p21, q12, q21)
        dt2 = dfxdt_2(xk, p12, p21, q12, q21)
        gk = array([dv1, dv2, dt2])
        g_result.append(numpy.linalg.norm(gk))
        k += 1

    plt.plot([i for i in range(k)], numpy.log(g_result))
    plt.xlabel("k")
    plt.ylabel("||dfx||")
    plt.savefig("P3 dfx vs k")
    plt.show()

    plt.plot([i for i in range(k)], numpy.log(x_result))
    plt.xlabel("k")
    plt.ylabel("||xk-x_true||^2")
    plt.savefig("P3 xk-x_true vs k")
    plt.show()

solve()
