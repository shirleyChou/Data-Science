# encoding: utf-8

import numpy as np


def load_data():
    data = np.genfromtxt("ex1data2.txt", delimiter=",")
    X, y = data[:, :2], data[:, 2]
    return X, y


def feature_normalization():
    X, y = load_data()
    m, n = X.shape
    means = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    x_normal = (X - means) / std
    return x_normal


def cost_function():
    x_normal = feature_normalization()

    m, n = x_normal.shape
    theta_initial =
    j_of_theta = () / (2 * m)


def gradient_descent():
    pass


def main():
    feature_normalization()


if __name__ == '__main__':
    main()