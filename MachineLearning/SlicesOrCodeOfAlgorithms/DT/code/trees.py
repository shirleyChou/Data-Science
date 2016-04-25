# encoding utf-8

from math import log

import pandas as pd


def calculate_empirical_entropy(dataset):
    number_of_Entries = len(dataset)
    class_of_labels = {}
    for element in dataset:
        label = element[-1]
        if label in class_of_labels.keys():
            class_of_labels[label] += 1
        else:
            class_of_labels[label] = 0
    empirical_entropy = 0.0
    for times in class_of_labels.values():
        percentage = times / number_of_Entries
        empirical_entropy += -1 * percentage * log(percentage, 2)
    return empirical_entropy


def calculate_empirical_conditional_entropy(dataset):
    pass


def calculate_information_gain(value1, value2):
    return value1 - value2


def read_data():
    data = pd.read_csv('data.csv')
    print data["Age"]


if __name__ == "__main__":
    read_data()
