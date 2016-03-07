# encoding: utf-8

import numpy as np
from matplotlib import pyplot as plt

def load_data():
    data = np.genfromtxt("ex1data2.txt", delimiter=",")
    X, y = data[:, :2], data[:, 2]
    return X, y


def feature_normalization(x):
    means = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_normal = (x - means) / std
    return means, std, x_normal,


def hypothesis(x, theta):
    return x.dot(theta.T)


def compute_cost(x, y, theta, m):
    h_x = hypothesis(x, theta)
    j_of_theta = (h_x - y).T.dot(h_x - y) / (2 * m)
    return j_of_theta[0]


def gradient_descent_multi(x, y, theta, alpha, iterations, m):
    grad = theta
    j_of_theta = []

    for i in xrange(iterations):
        j = compute_cost(x, y, grad, m)
        j_of_theta.append(j)
        h_x = hypothesis(x, grad)
        grad = grad - alpha / m * np.sum((h_x - y) * x, axis=0)
    return j_of_theta, grad


def normal_equation(x, y):
    theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return theta


def part3_1():
    x, y = load_data()
    mu, std, x_normal = feature_normalization(x)
    print mu, std


def part3_2():
    x, y = load_data()
    mu, std, x_normal = feature_normalization(x)
    m, n = x_normal.shape
    x = np.ones((m, n+1))
    x[:, 1:] = x_normal
    y = y.reshape((m, 1))

    alphas = [0.01, 0.03, 0.1, 0.3, 1.0]
    iterations = 400
    init_theta = np.ones((1, n+1))

    for alpha in alphas:
        j_of_theta, theta = gradient_descent_multi(x, y, init_theta,
                                                   alpha, iterations, m)
        print "Theta computed from gradient descent:\n %s" % theta
        plt.plot(range(iterations), j_of_theta, '-b')
        plt.title("Alpha = %s" % alpha)
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost J")
        plt.show(block=True)

        new = np.array([1650.0, 3.0])
        new_normal = (new - mu) / std
        new_add = np.ones((1, 3))
        new_add[:, 1:] = new_normal
        result = hypothesis(new_add, theta)
        print ("Predicted price of a 1650 sq-ft, 3 br house "
               "(using gradient descent):\n %s" % result)


def part3_3():
    x_raw, y = load_data()
    m, n = x_raw.shape
    x = np.ones((m, n+1))
    x[:, 1:] = x_raw
    y = y.reshape((m, 1))

    print "Solving with normal equations..."
    theta = normal_equation(x, y)
    print "Theta computed from the normal equations:\n %s" % theta

    new = np.array([1.0, 1650.0, 3.0])
    result = hypothesis(new, theta.T)
    print ("Predicted price of a 1650 sq-ft, 3 br house "
               "(using normal equations):\n %s" % result)


def main():
    # part3_1()
    # part3_2()
    part3_3()


if __name__ == '__main__':
    main()