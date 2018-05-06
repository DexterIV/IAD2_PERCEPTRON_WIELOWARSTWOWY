import random
import numpy

from DataStructure.NeuralNetwork import NeuralNetwork


def task_2():
    network = NeuralNetwork(4, 2, 4, 0.1, 0)

    input_list = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    output = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    epochs = 10000
    x = numpy.zeros(epochs * 4, dtype=int)

    for i in range(epochs):
        x[i] = 0
        x[i + epochs] = 1
        x[i + epochs * 2] = 2
        x[i + epochs * 3] = 3

    random.shuffle(x)

    for e in range(len(x)):
        network.train_manual_epochs(input_list[(x[e])], output[(x[e])])

    numpy.set_printoptions(suppress=True)  # avoid e-05 notation
    fin = [network.query(input_list[0]),
           network.query(input_list[1]),
           network.query(input_list[2]),
           network.query(input_list[3])]
    print(output)

    for elem in range(len(fin)):
        print(fin[elem])
