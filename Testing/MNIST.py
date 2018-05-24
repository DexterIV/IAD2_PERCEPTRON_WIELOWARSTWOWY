import copy
import random
import numpy
from library.funs import *
from mlp.NeuralNetwork import NeuralNetwork


def task_MNIST():
    numpy.set_printoptions(suppress=True)  # avoid e-05 notation
    hidden_nodes = 10
    epochs = 3
    learning_rate = 0.05
    momentum = 0.2
    bias = True
    network = NeuralNetwork(input_nodes=784, hidden_nodes=hidden_nodes, output_nodes=10,learning_rate=learning_rate,
                            momentum=momentum, bias=bias, epochs=epochs, error_sampling_rate=(1/epochs))

    # load the mnist training data CSV file into a list
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    # train the neural network
    # go through all records in the training data set
    for e in range(epochs):
        print("Epoch:" + str(e))
        collect_data_for_plot = True
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired    label    which is 0.99)
            targets = numpy.zeros(10) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            network.train_manual_epochs(inputs.tolist(), targets, e, collect_data_for_plot)
            collect_data_for_plot = False

    # load the mnist test data CSV file into a list
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    error = 0
    # test the neural network
    # scorecard for how well the network performs, initially empty
    scorecard = []
    # go through all the records in the test data set
    fin = []
    values = []
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        values.append(correct_label)
        # print(correct_label, "correct label")
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # query the network
        outputs = network.query(inputs.tolist())
        fin.append(outputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # print(label, "network's answer")
        # append correct or incorrect to list
        if label == correct_label:
            # network's answer matches correct answer, add 1 to        scorecard
            error += 1

    print("MNIST accuracy = " + str(error / 10000 * 100) + "%")

    parameters = parameters_as_string(hidden_nodes, learning_rate, momentum, epochs, bias)
    calculate_results_table (10, values, fin, 'MNIST result table\n' + parameters)
    print_plot(network.sampling_iteration, network.errors_for_plot, 'MNIST error plot \n' + parameters)
