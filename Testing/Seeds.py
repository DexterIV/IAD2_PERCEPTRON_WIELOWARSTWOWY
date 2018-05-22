import copy
import random
import numpy
from library.funs import *
from mlp.NeuralNetwork import NeuralNetwork


def task_Seeds():
    network = NeuralNetwork(7, 10, 3, 0.1, 0.1, bias=True)

    input_list = initialize_data_with1stcolumn("seeds_dataset.csv")
    output = []
    print(input_list)
    for i in range(int(len(input_list)/3)):
        output.append([[1, 0, 0], input_list[i]])
    for i in range(int(len(input_list)/3)):
        output.append([[0, 1, 0], input_list[i + int(len(input_list)/3)]])
    for i in range(int(len(input_list)/3)):
        output.append([[0, 0, 1], input_list[i + int(len(input_list)/3) * 2]])

    print(output)
    epochs = 5000

    shuffled_output = copy.copy(output)
    random.shuffle(shuffled_output)

    for i in range(epochs):
        for e in range(len(shuffled_output)):
            network.train_manual_epochs(shuffled_output[e][1], shuffled_output[e][0])

    fin = []
    numpy.set_printoptions(suppress=True)  # avoid e-05 notation
    for i in range(len(input_list)):
        fin.append(network.query(input_list[i]))
    # print(output)

    for elem in range(len(fin)):
        print(fin[elem])