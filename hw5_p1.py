from numpy import array
import numpy
import matplotlib.pyplot as plt


def fxy(vec):
    fxy = pow(vec[0], 2) - 5 * vec[0] * vec[1] + pow(vec[1], 4) - 25 * vec[0] - 8 * vec[1]
    return fxy


def df(vec):
    dx = 2 * vec[0] - 5 * vec[1] - 25
    dy = -5 * vec[0] + 4 * pow(vec[1], 3) - 8
    return array([dx, dy])


def hessian(vec):
    dxx = 2
    dxy = -5
    dyx = -5
    dyy = 12 * pow(vec[1], 2)
    return array([[dxx, dxy], [dyx, dyy]])


def get_alpha(vec, alpha, eta, e):
    h = numpy.linalg.inv(hessian(vec))
    g = df(vec)
    while fxy(vec - alpha * h.dot(g)) > (fxy(vec) - e * alpha * g.T.dot(h).dot(g)):
        alpha = alpha / eta
    return alpha


def get_e(e, vec_k, delta):
    eig_v, _ = numpy.linalg.eig(e * numpy.eye(2) + hessian(vec_k))
    lambda1 = numpy.min(eig_v)
    while e + lambda1 < delta:
        eig_v, _ = numpy.linalg.eig(e * numpy.eye(2) + hessian(vec_k))
        lambda1 = numpy.min(eig_v)
        if lambda1 >= delta:
            e = 0
        else:
            e = delta - lambda1
    return e


def newton():
    vec_0 = [1., 1.]
    k = 0
    e = 0.2
    a = 1.
    epsilon = 1e-4
    delta = 1
    eta = 2.
    dfx = []
    fx = []
    vec_k = vec_0
    while numpy.linalg.norm(df(vec_k)) > epsilon:
        e = get_e(e, vec_k, delta)
        a = get_alpha(vec_k, a, eta, e)
        vec_k = vec_k - a * numpy.linalg.inv((e * numpy.eye(2) + hessian(vec_k))).dot(df(vec_k))
        dfx.append(numpy.linalg.norm(df(vec_k)))
        fx.append(fxy(vec_k))
        k += 1

    plt.plot([i for i in range(k)], dfx)
    plt.xlabel("k")
    plt.ylabel("||dfx||")
    plt.savefig("h5 dfx vs k")
    plt.show()

    plt.plot([i for i in range(k)], fx)
    plt.xlabel("k")
    plt.ylabel("fx")
    plt.savefig("h5 fx vs k")
    plt.show()


newton()

