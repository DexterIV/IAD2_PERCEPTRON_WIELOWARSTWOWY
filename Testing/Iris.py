import copy
import random
import numpy
from library.funs import *
from mlp.NeuralNetwork import NeuralNetwork


def task_Iris():
    network = NeuralNetwork(4, 5, 3, 0.1, 0.1, bias=True)

    input_list = initialize_data("Iris.csv")
    output = []
    print(input_list)
    for i in range(int(len(input_list)/3)):
        output.append([[1, 0, 0], input_list[i]])
    for i in range(int(len(input_list)/3)):
        output.append([[0, 1, 0], input_list[i]])
    for i in range(int(len(input_list)/3)):
        output.append([[0, 0, 1], input_list[i]])

    print(output)
    epochs = 5000

    shuffled_output = copy.copy(output)
    random.shuffle(shuffled_output)

    for e in range(len(shuffled_output)):
        network.train_manual_epochs(output[e][1], output[e][0])

    fin = []
    numpy.set_printoptions(suppress=True)  # avoid e-05 notation
    for i in range(len(input_list)):
        fin.append(network.query(input_list[i]))
    # print(output)

    for elem in range(len(fin)):
        print(fin[elem])