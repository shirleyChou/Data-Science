# !/usr/bin/env python
# encoding: utf-8

from os.path import *
from math import exp

import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = join(dirname(__file__), 'ex2data1.txt')


def load_data():
    data = np.loadtxt(DATA_PATH, delimiter=',')
    X, y = data[:, :2], data[:, 2]
    return X, y


def plot_data(X, y):
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', s=40, marker='+',
                linewidth=2, label="Admitted")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='y', s=40, marker='o',
                linewidth=1, label="Not admitted")
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()


def sigmoid_func(z):
    g = 1.0 / (1 + np.exp(-z))
    return g


def cost_function(X, y, theta):
    h_theta = sigmoid_func(X.dot(theta))
    m, n = X.shape
    j_theta = np.sum(-y * np.log(h_theta) - (1 - y) * np.log(1 - h_theta)) / m
    grad = np.sum((h_theta - y) * X, axis=0) / m
    return j_theta, 3, grad

def part_1():
    print ('Plotting data with + indicating (y = 1) examples and o indicating '
           '(y = 0) examples')
    X, y = load_data()
    plot_data(X, y)
    plt.show(block=True)


def part_2():
    x, y = load_data()
    m, n = x.shape
    X = np.ones((m, n+1))
    X[:, 1:] = x
    y = y.reshape(m, 1)
    initial_theta = np.zeros((n+1, 1))
    cost, grad = cost_function(X, y, initial_theta)
    print 'Cost at initial theta (zeros): %s' % cost
    print 'Gradient at initial theta (zeros): \n %s' % grad


def part_3():
    pass


def main():
    # part_1()
    part_2()


if __name__ == '__main__':
    main()
