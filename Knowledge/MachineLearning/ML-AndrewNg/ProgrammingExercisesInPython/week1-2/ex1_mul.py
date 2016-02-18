# encoding: utf-8

import numpy as np


def load_data():
    data = np.genfromtxt("ex1data2.txt", delimiter=",")
    X, y = data[:, :2], data[:, 2]
    return X, y


def feature_normalization():
    X, y = load_data()
    means = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    x_normal = (X - means) / std
    return x_normal, y


def hypothesis(x, theta):
    return x.dot(theta)

def cost_function(x, y, theta):
    m = x.shape[0]
    h_x = hypothesis(x, theta)
    j_of_theta = (h_x - y).T.dot(h_x - y) / (2 * m)
    return j_of_theta


def gradient_descent_multi(x, y, theta, alpha, num_iters):
    h_x = hypothesis(x, theta)
    



def part_1():
    x_normal, y = feature_normalization()
    m, n = x_normal.shape
    x = np.ones((m, n+1))
    x[:, 1:] = x_normal
    initial_theta = np.ones((n+1, 1))
    alpha = 0.01
    num_iters = 400

    cost = cost_function(x, y.reshape((m, 1)), initial_theta)
    grad = gradient_descent_multi(x, y, initial_theta, alpha, num_iters)


def main():
    part_1()


if __name__ == '__main__':
    main()