from numpy import array
import numpy
import matplotlib.pyplot as plt


def f(x, Q, b):
    fx = 0.5 * x.T.dot(Q).dot(x) + b.T.dot(x)
    return fx


def g(x):
    gx = -1 * x
    return gx


def h(x):
    hx = pow(x[0], 2) + pow(x[1], 2) + pow(x[2], 2) + pow(x[3], 2) - 1
    return hx


def p(x):
    px = 0.5 * (pow(max(0, g(x)[0]), 2) + pow(max(0, g(x)[1]), 2) + pow(max(0, g(x)[2]), 2) + pow(max(0, g(x)[3]), 2)) + \
         0.5 * pow(h(x), 2)
    return px


def dq(x, Q, b, c):
    dgx = g(x)
    for i in range(len(dgx)):
        if dgx[i] <= 0:
            dgx[i] = 0
        else:
            dgx[i] = x[i]
    dpx = dgx + 2 * x * h(x)
    dqx = Q.dot(x) + b + c * dpx
    return dqx


def q(x, Q, b, c):
    qx = 0.5 * x.T.dot(Q).dot(x) + b.T.dot(x) + c * p(x)
    return qx


def get_alpha(x, alpha, eta, e, Q, b, c):
    while q(x - alpha * dq(x, Q, b, c), Q, b, c) > (q(x, Q, b, c) - e * alpha * dq(x, Q, b, c).T.dot(dq(x, Q, b, c))):
        alpha = alpha / eta
    return alpha


def partan(x0, Q, b, c):
    alpha_partan = 1e-4
    beta_partan = 1e-4

    eta_partan = 2.
    e_partan = 0.2

    alpha_partan = get_alpha(x0, alpha_partan, eta_partan, e_partan, Q, b, c)
    x1 = x0 - alpha_partan * dq(x0, Q, b, c)

    xk_minus_1 = x0
    xk = x1
    gk = dq(xk, Q, b, c)

    while numpy.linalg.norm(gk) > 1e-3:
        alpha_partan = get_alpha(xk, alpha_partan, eta_partan, e_partan, Q, b, c)
        beta_partan = get_alpha(xk, beta_partan, eta_partan, e_partan, Q, b, c)

        xk_plus_1 = xk - alpha_partan * (1 + beta_partan) * gk + beta_partan * (xk - xk_minus_1)

        xk_minus_1 = xk
        xk = xk_plus_1
        gk = dq(xk, Q, b, c)
        # print(numpy.linalg.norm(gk))
    return xk


def penalty():
    Q = array([[2., 1., 0., 10.], [1., 4., 3., .5], [0., 3., -5., 6.], [10., .5, 6., -7.]])
    b = array([-1., 0., -2., 3.])
    c = 10.
    epsilon = 1e-3
    beta = 2.

    fx_list = []
    c_p_list = []
    p_list = []
    q_list = []

    # x0 = array([1, 1, 1, 1])
    x0 = numpy.random.random(4)
    xk = x0
    while c * p(xk) > epsilon:
        xk = partan(xk, Q, b, c)
        c = beta * c

        fx_list.append(f(xk, Q, b))
        c_p_list.append(c * p(xk))
        p_list.append(p(xk))
        # q_list.append(q(xk, Q, b, c))
    # print(xk)
    plt.plot([x for x in range(len(fx_list))], fx_list)
    plt.xlabel("k")
    plt.ylabel("f(xk)")
    plt.savefig("final_fxk")
    plt.show()
    plt.plot([x for x in range(len(c_p_list))], c_p_list)
    plt.xlabel("k")
    plt.ylabel("c*p(xk)")
    plt.savefig("final_cpxk")
    plt.show()
    plt.plot([x for x in range(len(p_list))], p_list)
    plt.xlabel("k")
    plt.ylabel("p(xk)")
    plt.savefig("final_p(xk)")
    plt.show()
    # plt.plot([x for x in range(len(q_list))], q_list)
    # plt.xlabel("k")
    # plt.ylabel("q(xk)")
    # plt.savefig("final_q(xk)")
    # plt.show()

    check_kkt(xk, Q, b)


def check_kkt(x, Q, b):
    opt = x       # check KKT
    A = array([[-1, -1, -1, -1, 2*(opt[0]+opt[1]+opt[2]+opt[3])]]).T
    df = array([-(Q.dot(opt) + b).T])

    lambda_mu = numpy.linalg.pinv(A.dot(A.T)).dot(A).dot(df)
    dfx = array([Q.dot(opt) + b]).T
    result = dfx + lambda_mu.T.dot(A)
    print(result.round(12))


penalty()
