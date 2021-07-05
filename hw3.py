from numpy import array
import numpy
from qpsolvers import solve_qp
import matplotlib.pyplot as plt


Q = array([[1., .5, 0.], [.5, 1, .25], [0., .25, 1.]])
a = array([1., -1., 2.])


def pa():
    eig_val, eig_vec = numpy.linalg.eig(Q)
    print("pa: max:", max(eig_val))


def pb():
    x = solve_qp(Q, a)
    print("pb: QP solution: x = {}".format(x))
    return x


def pc():
    x = pb()
    x0 = numpy.random.random(3)     # set arbitrary x0
    print(x0)
    alpha = 1e-3        # set alpha
    gk = 0.5 * Q.dot(x0) + 0.5 * Q.T.dot(x0) + a
    epsilon = 1e-7      # set epsilon
    xk = x0
    x_list = []
    while numpy.linalg.norm(gk) > epsilon:
        xk = xk - alpha * gk
        gk = 0.5 * Q.dot(xk) + 0.5 * Q.T.dot(xk) + a
        x_list.append(numpy.linalg.norm(xk-x))
    plt.plot([x for x in range(len(x_list))], numpy.log(x_list))
    plt.xlabel("k")
    plt.ylabel("||xk-x*||")
    plt.savefig("pc")
    plt.show()


# pa()
# pb()
pc()

