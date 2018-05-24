import copy
import random
import numpy
from library.funs import *
from mlp.NeuralNetwork import NeuralNetwork


def task_MNIST():
    network = NeuralNetwork(784, 140, 10, 0.1, 0.2, bias=True)

    # load the mnist training data CSV file into a list
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    # train the neural network
    epochs = 3
    # go through all records in the training data set
    for e in range(epochs):
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired    label    which is 0.99)
            targets = numpy.zeros(10) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            network.train_manual_epochs(inputs.tolist(), targets, e)

    # load the mnist test data CSV file into a list
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    error = 0
    # test the neural network
    # scorecard for how well the network performs, initially empty
    scorecard = []
    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        print(correct_label, "correct label")
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = network.query(inputs.tolist())
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        print(label, "network's answer")
        # append correct or incorrect to list
        if label == correct_label:
            # network's answer matches correct answer, add 1 to        scorecard
            error += 1

    print("error for seeds = " + str(error / 10000 * 100) + "%")

    funs.print_plot(network.sampling_iteration, network.errors_for_plot, 'MNIST error plot')