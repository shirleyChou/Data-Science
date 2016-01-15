# !/usr/bin/env python
# encoding: utf-8

import os

import numpy as np
from matplotlib import pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), 'ex1data1.txt')


def part1():
    print 'Running warmUpExercise ... '
    print '5x5 Identity Matrix: '
    print np.eye(5)


def hypothesis(X, theta):
    return np.dot(X, theta)


def plot_data(X, y):
    print 'Plotting Data ...'
    plt.plot(X, y, 'rx', markersize=5)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.xlim(4, 25)
    plt.ylim(-5, 25)


def compute_cost(X, y, theta):
    h_x = hypothesis(X, theta)
    m = len(y)
    return round(np.sum((h_x - y)**2) / (2 * m), 2)


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    grad = theta

    for i in xrange(iterations):
        inner_sum = X.T.dot(hypothesis(X, grad) - y)
        grad = grad - alpha / m * inner_sum
    return grad


def load_data():
    data = np.loadtxt(DATA_PATH, delimiter=',')
    X, y = data[:, 0], data[:, 1]
    return X, y


def part2_1():
    X, y = load_data()
    plot_data(X, y)
    plt.show(block=True)


def part2_2():
    x, y = load_data()
    m = len(y)
    x, y = x.reshape(m, 1), y.reshape(m, 1)
    X = np.ones((x.shape[0], x.shape[1]+1))
    X[:, 1:] = x
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.02

    cost = compute_cost(X, y, theta)   # the cost should be 32.07
    theta = gradient_descent(X, y, theta, alpha, iterations)
    print 'ans= ', cost
    print 'Theta found by gradient descent: ', float(theta[0]), float(theta[1])

    predict1 = np.array([1, 3.5]).dot(theta) * 10000
    predict2 = np.array([1, 7]).dot(theta) * 10000
    print 'For population = 35,000, we predict a profit of %s' % float(predict1)
    print 'For population = 70,000, we predict a profit of %s' % float(predict2)

    plot_data(X[:, 1], y)
    plt.plot(X[:, 1], X.dot(theta), 'b-')
    plt.show(block=True)


def part2_4():
    print 'Visualizing J(theta_0, theta_1) ...'
    x, y = load_data()
    m = len(y)
    x, y = x.reshape(m, 1), y.reshape(m, 1)
    X = np.ones((x.shape[0], x.shape[1]+1))
    X[:, 1:] = x

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    j_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i in xrange(len(theta0_vals)):
        for j in xrange(len(theta1_vals)):
            t = np.array((theta0_vals[i], theta1_vals[j])).reshape(2, 1)
            j_vals[i, j] = compute_cost(X, y, t)

    R, P = np.meshgrid(theta0_vals, theta1_vals)

    # fig = plt.figure()
    # ax 	= fig.gca()
    # ax.contourf(R, P, j_vals)
    # plt.show(block=True)

    # plt.contourf(R, P, j_vals.T, np.logspace(-2, 3, 20))
    # plt.plot(t[0], t[1], 'rx', markersize = 10)
    # plt.show(block=True)




def main():
    part1()
    part2_1()
    part2_2()
    part2_4()

if __name__ == '__main__':
    main()
