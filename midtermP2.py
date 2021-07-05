from numpy import array
import numpy
from qpsolvers import solve_qp
import matplotlib.pyplot as plt

Q = array([[1., .5, 0.], [.5, 1., .25], [0., .25, 1.]])
b = array([-1., 1., -2.])


def gk(x):  # define gx
    gk = Q.dot(x) - b
    return gk


def alpha(gk):  # define alpha
    alpha = (gk.T.dot(gk)) / (gk.T.dot(Q).dot(gk))
    return alpha


def ex(x, x_star):  # define ex
    ex = 0.5 * (x - x_star).T.dot(Q).dot(x-x_star)
    return ex


def solve():
    x0 = numpy.random.random(3)   # get a arbitrary x0
    result = []
    result_ex = []
    k = 0   # set k = 0
    x_star = solve_qp(Q, -b)
    epsilon = 1e-6
    xk = x0
    while numpy.linalg.norm(gk(xk)) > epsilon:  # while criterion is satisfied
        g_list = gk(xk)
        index = numpy.argmax(numpy.absolute(g_list))
        g_bar = [0., 0., 0.]
        g_bar[index] = g_list[index]
        a = alpha(numpy.array(g_bar))
        xk = xk - a * numpy.array(g_bar)
        ex_val = ex(xk, x_star)
        result_ex.append(ex_val)
        k += 1
        result.append(numpy.linalg.norm(gk(xk)))
    plt.plot([x for x in range(k)], numpy.log(result))
    plt.xlabel("k")
    plt.ylabel("||gx||")
    plt.savefig("P2 gk vs k")
    plt.show()

    plt.plot([x for x in range(k)], numpy.log(result_ex))
    plt.xlabel("k")
    plt.ylabel("ex")
    plt.savefig("P2 fx vs k")
    plt.show()


solve()

