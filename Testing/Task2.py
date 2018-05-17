import random
import numpy

from mlp.NeuralNetwork import NeuralNetwork


def task_2():
    network = NeuralNetwork(input_nodes=4, output_nodes=4, hidden_layers_quantity=1, hidden_nodes=2,
                            learning_rate=0.1, momentum=0.0, bias=False)

    input_list = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    output = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    epochs = 50000
    x = numpy.zeros(epochs * 4, dtype=int)
    for i in range(epochs):
        x[i] = 0
        x[i + epochs] = 1
        x[i + epochs * 2] = 2
        x[i + epochs * 3] = 3

    random.shuffle(x)

    for e in range(len(x)):
        network.train_manual_epochs(input_list[(x[e])], output[(x[e])], 0.1)

    numpy.set_printoptions(suppress=True)  # avoid e-05 notation
    fin = [network.query(input_list[0]),
           network.query(input_list[1]),
           network.query(input_list[2]),
           network.query(input_list[3])]
    print(output)

    for elem in range(len(fin)):
        print(fin[elem])
