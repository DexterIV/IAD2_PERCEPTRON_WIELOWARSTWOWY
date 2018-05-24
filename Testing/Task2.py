import random
import numpy
from library import funs

from mlp.NeuralNetwork import NeuralNetwork

def task_2():
    epochs = 5000
    network = NeuralNetwork(4, 2, 4, 0.1, 0.1, bias=True, epochs=20000)

    input_list = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    output = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]

    x = numpy.zeros(epochs * 4, dtype=int)
    for i in range(epochs):
        x[i] = 0
        x[i + epochs] = 1
        x[i + epochs * 2] = 2
        x[i + epochs * 3] = 3

    random.shuffle(x)

    for e in range(len(x)):
        network.train_manual_epochs(input_list[(x[e])], output[(x[e])], e)

    fin = []
    query_list = []
    for i in range(64):
        query_list.append(input_list[x[i]])
        fin.append(network.query(query_list[i]))

    numpy.set_printoptions(suppress=True)  # avoid e-05 notation
    print(output)

    for elem in range(len(fin)):
        print(fin[elem])

    funs.calculate_results_table(len(input_list), query_list, fin)
    funs.print_plot(network.sampling_iteration, network.errors_for_plot, 'Task 2 error plot')
