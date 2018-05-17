import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return x * (1 - x)


def initialize_data(filename):
    data = []
    dataLabels = []
    dataset_tmp = pd.read_csv(filename, header=None)
    number_of_columns = len(dataset_tmp.columns)
    data_attributes = []

    for i in range(1, number_of_columns - 1):
        dataLabels.append(dataset_tmp[i][0])

    dataset = pd.read_csv(filename)

    for i in range(1, number_of_columns - 1):
        data_attributes.append(dataset.iloc[:, i].values)

    for i in range(1, len(data_attributes[0])):
        values = []
        for j in range(len(data_attributes)):
            values.append(data_attributes[j][i])
        data.append(values)

    return data


