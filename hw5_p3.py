from numpy import array
import numpy
import matplotlib.pyplot as plt
import scipy.io


def fx(x, Q, b, c, mu):
    fx = 0.5 * x.T.dot(Q).dot(x) - b.T.dot(x) + 0.5 * mu * pow(c.T.dot(x), 2)
    return fx


def dfx(x, Q, b, c, mu):
    dfx = Q.dot(x) - b + mu * x.T.dot(c) * c
    return dfx


def data():
    data = scipy.io.loadmat("HW5_data.mat")
    Q = array(data['Q'])
    b = array(data['b'])
    c = array(data['c'])
    return Q, b, c


def hessian(Q, mu, c):
    hessian = Q + mu * c.dot(c.T)
    return hessian


def pa():
    Q, b, c = data()
    x0 = array([numpy.random.random(1000)]).T
    alpha = 1e-6
    # mu = 1
    mu = 1000
    gk = Q.dot(x0) - b + mu * x0.T.dot(c) * c
    epsilon = 1e-4
    xk = x0
    x_list = []
    hes = hessian(Q, mu, c)
    eig_v, _ = numpy.linalg.eig(hes)
    while numpy.linalg.norm(gk) > epsilon:
        xk = xk - alpha * gk
        gk = dfx(xk, Q, b, c, mu)
        x_list.append(numpy.linalg.norm(gk))

    plt.plot([x for x in range(len(x_list))], x_list)
    plt.xlabel("k")
    plt.ylabel("||gk||")
    plt.savefig("hw5p3a")
    plt.show()
    return xk


def get_alpha(x, alpha, eta, e, Q, b, c, mu):
    while fx(x + alpha * -dfx(x, Q, b, c, mu), Q, b, c, mu) > (fx(x, Q, b, c, mu) + e * alpha * dfx(x, Q, b, c, mu).T.dot(-dfx(x, Q, b, c, mu))):
        alpha = alpha / eta
    return alpha


def pb():
    x = pa()
    Q, b, c = data()
    x0 = array([numpy.zeros(1000)]).T
    alpha = 1e-6
    # mu = 1
    mu = 1000
    eta = 2.
    e = 0.2
    gk = Q.dot(x0) - b + mu * x0.T.dot(c) * c

    epsilon = 1e-4
    xk = x0
    dfx_list = []
    fx_list = []
    x_list = []

    while numpy.linalg.norm(gk) > epsilon:
        alpha = get_alpha(xk, alpha, eta, e, Q, b, c, mu)
        xk = xk - alpha * gk
        gk = Q.dot(xk) - b + mu * xk.T.dot(c) * c
        dfx_list.append(numpy.linalg.norm(gk))
        fx_list.append(fx(xk, Q, b, c, mu)[0, 0])
        x_list.append(numpy.linalg.norm(x-xk))

    plt.plot([i for i in range(len(dfx_list))], dfx_list)
    plt.xlabel("k")
    plt.ylabel("||gk||")
    plt.savefig("hw5p3bgk-u1000")
    plt.show()

    plt.plot([i for i in range(len(fx_list))], fx_list)
    plt.xlabel("k")
    plt.ylabel("fx")
    plt.savefig("hw5p3bfx-u1000")
    plt.show()

    plt.plot([i for i in range(len(x_list))], x_list)
    plt.xlabel("k")
    plt.ylabel("||xk-x||")
    plt.savefig("hw5p3bxk-u1000")
    plt.show()


def pc():
    x = pa()
    Q, b, c = data()
    alpha = 1e-6
    beta = 2e-6
    # mu = 1
    mu = 1000
    eta = 2.
    e = 0.2
    dfx_list = []
    fx_list = []
    x_list = []

    x0 = array([numpy.zeros(1000)]).T
    x1 = x0 - alpha * dfx(x0, Q, b, c, mu)
    epsilon = 1e-4

    xk_minus_1 = x0
    xk = x1
    gk = dfx(x1, Q, b, c, mu)
    while numpy.linalg.norm(gk) > epsilon:
        alpha = get_alpha(xk, alpha, eta, e, Q, b, c, mu)
        beta = get_alpha(xk, beta, eta, e, Q, b, c, mu)
        xk_plus_1 = xk - alpha * (1 + beta) * gk + beta * (xk - xk_minus_1)
        dfx_list.append(numpy.linalg.norm(gk))
        fx_list.append(fx(xk, Q, b, c, mu)[0, 0])
        x_list.append(numpy.linalg.norm(x - xk))
        xk_minus_1 = xk
        xk = xk_plus_1
        gk = Q.dot(xk) - b + mu * xk.T.dot(c) * c

    plt.plot([i for i in range(len(dfx_list))], dfx_list)
    plt.xlabel("k")
    plt.ylabel("||gk||")
    plt.savefig("hw5p3cgk-u1000")
    plt.show()

    plt.plot([i for i in range(len(fx_list))], fx_list)
    plt.xlabel("k")
    plt.ylabel("fx")
    plt.savefig("hw5p3cfx-u1000")
    plt.show()

    plt.plot([i for i in range(len(x_list))], x_list)
    plt.xlabel("k")
    plt.ylabel("||xk-x||")
    plt.savefig("hw5p3cxk-u1000")
    plt.show()

# pa()
# pb()
pc()



