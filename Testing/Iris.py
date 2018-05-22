import copy
import random
import numpy
from library.funs import *
from mlp.NeuralNetwork import NeuralNetwork


def task_Iris():
    network = NeuralNetwork(input_nodes=4, hidden_nodes=10,
                            output_nodes=3, learning_rate=0.005,
                            momentum=0.1, bias=True)

    input_list = initialize_data("Iris.csv")
    output = []
    indices = []
    print(input_list)
    for i in range(int(len(input_list) / 3)):
        output.append([[1, 0, 0], input_list[i]])
        indices.append(0)
    for i in range(int(len(input_list) / 3)):
        output.append([[0, 1, 0], input_list[i + int(len(input_list) / 3)]])
        indices.append(1)
    indices.append(1)
    for i in range(int(len(input_list) / 3)):
        output.append([[0, 0, 1], input_list[i + int(len(input_list) / 3) * 2]])
        indices.append(2)
    indices.append(2)

    print(output)
    epochs = 500

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

    error = 0
    for elem in range(len(fin)):
        if numpy.argmax(fin[elem]) != indices[elem]:
            error += 1

    print("error = " + str(error / len(fin) * 100) + "%")
