import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot


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

def initialize_data_with1stcolumn(filename):
    data = []
    dataLabels = []
    dataset_tmp = pd.read_csv(filename, header=None)
    number_of_columns = len(dataset_tmp.columns)
    data_attributes = []

    for i in range(0, number_of_columns - 1):
        dataLabels.append(dataset_tmp[i][0])

    dataset = pd.read_csv(filename)

    for i in range(0, number_of_columns - 1):
        data_attributes.append(dataset.iloc[:, i].values)

    for i in range(0, len(data_attributes[0])):
        values = []
        for j in range(len(data_attributes)):
            values.append(data_attributes[j][i])
        data.append(values)

    return data


def print_plot(x_axis, y_axis, title):
    pyplot.figure(title)
    pyplot.plot(x_axis, y_axis, 'r-', linestyle='solid', linewidth=1)
    pyplot.grid(axis='both', color='black', which='major', linestyle='--', linewidth=1)
    pyplot.xlabel("epoch")
    pyplot.ylabel("error")
    pyplot.suptitle(title)
    pyplot.show()


def get_index_of_max_element(source):
    index_max = 0
    for i in range(1, len(source)):
        if source[i] > source[index_max]:
            index_max = i
    return index_max


def calculate_results_table(number_of_classes, expected, actual):
    result_tab = np.zeros(shape=(number_of_classes, number_of_classes))

    for i in range(len(actual)):
        result_tab[get_index_of_max_element(expected[i])][get_index_of_max_element(actual[i])] += 1

    print('printing results table')
    print(result_tab)
