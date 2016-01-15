# !/usr/bin/env python
# encoding: utf-8

from os.path import *

import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = join(dirname(__file__), 'ex2data1.txt')


def load_data():
    data = np.loadtxt(DATA_PATH, delimiter=',')
    X, y = data[:, :2], data[:, 2]
    return X, y


def plot_data(X, y):
    plt.plot(X, y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')


def part_1():
    print ('Plotting data with + indicating (y = 1) examples and o indicating '
           '(y = 0) examples')
    X, y = load_data()
    plot_data(X, y)
    plt.show(block=True)


def main():
    part_1()


if __name__ == '__main__':
    main()