import numpy
from numpy import array
import matplotlib.pyplot as plt


alpha = 1
epsilon = 0.3
eta = 2
x, y = 0, 0


def f(x, y):
    fxy = pow(x, 2) - 5*x*y + pow(y, 4) - 25*x - 8*y
    return fxy


def df(x, y):
    dxfxy = 2*x - 5*y - 25
    dyfxy = 4 * pow(y, 3) - 5 * x - 8
    return dxfxy, dyfxy


def armijo(x, y, alpha):
    grad_x, grad_y = df(x, y)
    if (f(x + alpha * (-grad_x), y + alpha * (-grad_y))) <= (f(x, y) + epsilon * alpha * -1 * (grad_x * grad_x + grad_y * grad_y)) and (f(x + alpha * eta * (-grad_x), y + alpha * eta * (-grad_y))) <= (f(x, y) + epsilon * alpha * eta * -1 * (grad_x * grad_x + grad_y * grad_y)):
        while (f(x + alpha * (-grad_x), y + alpha * (-grad_y))) <= (f(x, y) + epsilon * alpha * -1 * (grad_x * grad_x + grad_y * grad_y)) and (f(x + alpha * eta * (-grad_x), y + alpha * eta * (-grad_y))) <= (f(x, y) + epsilon * alpha * eta * -1 * (grad_x * grad_x + grad_y * grad_y)):
            alpha = alpha * eta

    if (f(x + alpha * (-grad_x), y + alpha * (-grad_y))) > (f(x, y) + epsilon * alpha * -1 * (grad_x * grad_x + grad_y * grad_y)):
        while (f(x + alpha * (-grad_x), y + alpha * (-grad_y))) > (f(x, y) + epsilon * alpha * -1 * (grad_x * grad_x + grad_y * grad_y)):
            alpha = alpha / eta
    return alpha


def goldstein(x, y, alpha, epsilon):
    grad_x, grad_y = df(x, y)
    while (f(x + alpha * (-grad_x), y + alpha * (-grad_y))) > (f(x, y) + epsilon * alpha * -1 * (grad_x * grad_x + grad_y * grad_y)):
        alpha = alpha / eta
    return alpha


def decent_pa(x, y, alpha):
    res = f(x, y)
    rs = numpy.inf
    list_xk = []
    list_res = []
    while res < rs - 1e-10:
        rs = res
        grad_x, grad_y = df(x, y)
        list_xk.append(numpy.linalg.norm(grad_x))
        alpha = armijo(x, y, alpha)
        x -= alpha * grad_x
        y -= alpha * grad_y
        res = f(x, y)
        list_res.append(res)

    plt.plot([x for x in range(len(list_xk))], list_xk, color='blue')
    plt.xlabel("k")
    plt.ylabel("||df(xk)||")
    plt.savefig("pa1")
    plt.show()

    plt.plot([x for x in range(len(list_res))], list_res, color='red')
    plt.xlabel("k")
    plt.ylabel("f(xk)")
    plt.savefig("pa2")
    plt.show()
    hessian()


def decent_pb(x, y, alpha, epsilon):
    res = f(x, y)
    rs = numpy.inf
    list_xk = []
    list_res = []
    while res < rs - 1e-10:
        rs = res
        grad_x, grad_y = df(x, y)
        list_xk.append(numpy.linalg.norm(grad_x))
        alpha = goldstein(x, y, alpha, epsilon)
        x -= alpha * grad_x
        y -= alpha * grad_y
        res = f(x, y)
        list_res.append(res)
    print(x, y, res)

    plt.plot([x for x in range(len(list_xk))], list_xk, color='blue')
    plt.xlabel("k")
    plt.ylabel("||df(xk)||")
    plt.savefig("pb1")
    plt.show()

    plt.plot([x for x in range(len(list_res))], list_res, color='red')
    plt.xlabel("k")
    plt.ylabel("f(xk)")
    plt.savefig("pb2")
    plt.show()
    hessian()

def hessian():
    M = array([[2., -5.], [-5., 108.]])
    eig_val, eig_vec = numpy.linalg.eig(M)

decent_pa(x, y, alpha)
decent_pb(x, y, alpha, epsilon)
