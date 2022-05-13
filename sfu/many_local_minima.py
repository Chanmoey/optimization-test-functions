"""
这些最优化测试函数来自于: www.sfu.ca/~ssurjano/optimization.html。
对《Many Local Minima》这部分的实现。

依赖的库: numpy
"""
import numpy as np


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    """
    Dimensions: d
    xi ∈ [-32.768, 32.768]
    min f(x) = 0, at x = (0, 0, ..., 0)
    :param x: 函数参数
    :param a: 推荐20
    :param b: 推荐0.2
    :param c: 推荐 2Π
    :return:  函数值
    """
    d = len(x)
    sum1 = 0.0
    sum2 = 0.0
    for i in range(d):
        xi = x[i]
        sum1 += np.power(xi, 2)
        sum2 += np.cos(c * xi)
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)


def bukin(x):
    """
    Dimensions: 2
    The function is usually evaluated on the rectangle x1 ∈ [-15, -5], x2 ∈ [-3, 3].
    min f(x) = 0, at x = (-10, 1)
    :param x: 函数参数
    :return:  函数值
    """
    x1 = x[0]
    x2 = x[1]
    term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * np.power(x1, 2)))
    term2 = 0.01 * np.abs(x1 + 10)
    return term1 + term2


def cross_in_tray(x):
    """
    Dimensions: 2
    The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.
    min f(x) = -2.06261, at x = (+-1.3491, +-1.3491)
    :param x: 函数参数
    :return:  函数值
    """
    x1 = x[0]
    x2 = x[1]
    fact1 = np.sin(x1) * np.sin(x2)
    fact2 = np.exp(np.abs(100 - np.sqrt(np.power(x1, 2) + np.power(x2, 2)) / np.pi))
    return -0.0001 * np.power(np.abs(fact1 * fact2) + 1, 0.1)


def drop_wave(x):
    """
    Dimensions: 2
    The function is usually evaluated on the square xi ∈ [-5.12, 5.12], for all i = 1, 2.
    min f(x) = -1, at x = (0, 0)
    :param x: 函数参数
    :return:  函数值
    """
    x1 = x[0]
    x2 = x[1]
    frac1 = 1.0 + np.cos(12 * np.sqrt(np.power(x1, 2) + np.power(x2, 2)))
    frac2 = 0.5 * (np.power(x1, 2) + np.power(x2, 2)) + 2
    return -frac1 / frac2


def eggholder(x):
    """
    Dimensions: 2
    The function is usually evaluated on the square xi ∈ [-512, 512], for all i = 1, 2.
    min f(x) = -959.6407, at x = (512, 404.2319)
    :param x: 函数参数
    :return:  函数值
    """
    x1 = x[0]
    x2 = x[1]
    term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    return term1 + term2