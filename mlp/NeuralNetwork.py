import numpy
from library import funs
import copy


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, momentum, bias, hidden_layers_quantity):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.initial_learning_rate = learning_rate
        self.momentum = momentum
        self.activation_function = lambda x: funs.sigmoid(x)
        self.activation_function_derivative = lambda x: funs.sigmoid_der(x)
        self.hidden_layers_quantity = hidden_layers_quantity
        self.layers = self._create_hidden_layers()
        self.bias = bias
        if bias:
            self.biases = numpy.random.uniform(size=(hidden_layers_quantity + 2, hidden_nodes))
        else:
            self.biases = numpy.zeros(shape=(hidden_layers_quantity + 2, hidden_nodes))

    def _create_hidden_layers(self):
        layers = [numpy.random.uniform(size=(self.hidden_nodes, self.input_nodes))]
        # skip first layer since its initialized already
        for i in range(1, self.hidden_layers_quantity):
            # weights between hidden layers
            layers.append(numpy.random.uniform(size=(self.hidden_nodes, self.hidden_nodes)))
        # weigth from last hidden layer to output layer
        layers.append(numpy.random.uniform(size=(self.output_nodes, self.hidden_nodes)))
        return numpy.array(layers)

    def _learning_rate_decay(self, iteration):
        import math
        return self.initial_learning_rate * math.exp(- iteration)

    def train(self, input_list, target_list, epochs):
        learning_rate = self.initial_learning_rate
        for e in range(epochs):
            self.train_manual_epochs(input_list, target_list, learning_rate)
            # learning_rate = self._learning_rate_decay(e)

    def train_manual_epochs(self, input_list, target_list, learning_rate):
        # prepare array for neuron outputs
        neuron_outputs = []

        # convert inputs list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        # calculate signals into hidden layer
        neuron_outputs.append(numpy.dot(self.layers[0], inputs))

        for i in range(1, len(self.layers)):
            tmp = self.activation_function (neuron_outputs[i-1])
            neuron_outputs.append(numpy.dot(self.layers[i],
                                            self.activation_function(neuron_outputs[i - 1])))

        neuron_outputs = numpy.array(neuron_outputs)
        # initialize list of errors and insert output layer error (i ­ actual)
        neuron_errors = [targets - neuron_outputs[len(neuron_outputs) - 1]]  # take last outputs

        # insert in front since its iterating backwards
        for i in range(len(self.layers) - 2, -1, -1):
            neuron_errors.insert(0, numpy.dot(self.layers[i], neuron_errors[0]))

        # iterate from last layer to first
        for i in range(len(self.layers) - 1, -1, -1):
            self.layers[i] += (2 * learning_rate * numpy.dot(
                self.activation_function_derivative(neuron_outputs[i].T),
                neuron_errors[i])) * (1 + self.momentum)
            if self.bias:
                self.biases[i] += (2 * learning_rate * numpy.dot(
                    self.activation_function_derivative(neuron_outputs[i].T),
                    neuron_errors[i])) * (1 + self.momentum)

    def add_by_inedxes(self, target, damn):
        for i in range(len(target)):
            target[i] += damn[i]

    def query(self, input_list):
        # prepare array for neuron outputs
        neuron_outputs = []
        # convert inputs list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        # calculate signals into hidden layer
        neuron_outputs.append(numpy.dot(self.layers[0], inputs))

        for i in range(1, len (self.layers)):
            neuron_outputs.append(numpy.dot(self.layers[i],
                                            self.activation_function(neuron_outputs[i - 1])))

        return neuron_outputs[len(neuron_outputs) - 1]
