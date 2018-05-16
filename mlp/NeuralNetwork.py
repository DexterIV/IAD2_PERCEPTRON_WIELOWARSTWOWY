import numpy
from library import funs
import copy

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, momentum, bias, hidden_layers_quantity):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.bias = bias
        self.hidden_layers_quantity = hidden_layers_quantity
        self.hidden_layers = self._create_hidden_layers()
        self.activation_function = lambda x: funs.sigmoid(x)
        self.activation_function_derivative = lambda x: funs.sigmoid_der(x)

    def _create_hidden_layers(self):
        layers = []
        i = 1
        if self.bias:
            # weight input to hidden with bias, so + 1
            self.first_layer_weights = (numpy.random.rand (self.hidden_nodes, self.input_nodes + 1))
            # create bias for the first layer
            self.first_layer_weights = numpy.vstack([self.first_layer_weights, numpy.zeros(self.input_nodes + 1)])
            # layers.append(wih)
            # create one less, because first hidden layer has been already initialized
            for i in range(1, self.hidden_layers_quantity):
                # weight of previous layer on next
                tmp = numpy.random.rand(self.hidden_nodes, self.hidden_nodes)
                tmp = numpy.vstack([tmp, numpy.zeros(self.input_nodes + 1)])
                layers.append(tmp)
        else:
            # weight input to hidden without bias
            self.first_layer_weights = (numpy.random.rand(self.hidden_nodes, self.input_nodes))
            # layers.append(wih)
            # create one less, because first hidden layer has been already initialized
            for i in range(1, self.hidden_layers_quantity):
                # weight of previous layer on next
                layers.append(numpy.random.rand(self.hidden_nodes, self.hidden_nodes))  # weight input to hidden

        # weight from last hidden layer to output
        # if layers list is empty use wih instead
        if layers:
            self.output_weights = numpy.random.rand(self.output_nodes, len(layers[i - 1]))
        else:
            self.output_weights = numpy.random.rand(self.output_nodes, len(self.first_layer_weights))
        return numpy.array(layers)

    def train(self, input_list, target_list, epochs):
        for e in range(epochs):
            self.train_manual_epochs(input_list, target_list)

    def train_manual_epochs(self, input_list, target_list):
        # prepare array for neuron outputs
        neuron_outputs = []

        # convert inputs list to 2d array
        if self.bias:
            temp = copy.copy(input_list)
            temp.append(1)
            inputs = numpy.array(temp, ndmin=2).T
            targets = numpy.array(target_list, ndmin=2).T
            # calculate signals into hidden layer
            neuron_outputs.append(numpy.dot(self.first_layer_weights, inputs))
        else:
            inputs = numpy.array(input_list, ndmin=2).T
            targets = numpy.array(target_list, ndmin=2).T
            # calculate signals into hidden layer
            neuron_outputs.append(numpy.dot(self.first_layer_weights, inputs))

        for i in range(1, len(self.hidden_layers)):
            neuron_outputs.append(numpy.dot(self.hidden_layers[i], self.activation_function(neuron_outputs[i - 1])))

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(numpy.dot(self.output_weights,
                                                           self.activation_function
                                                           (neuron_outputs[len(neuron_outputs) - 1])))

        neuron_outputs = numpy.array(neuron_outputs)
        # initialize list of errors and insert output layer error (i ­ actual)
        neuron_errors = [targets - final_outputs]

        for i in range(len(self.hidden_layers), 0, -1):
            neuron_errors.insert(0, numpy.dot(self.hidden_layers[i].T, neuron_errors[0]))

        first_layer_errors = numpy.dot(self.first_layer_weights, neuron_errors[0])

        # backpropagation
        self.output_weights += (self.learning_rate * numpy.dot(
            (neuron_errors[len(neuron_errors) - 1] * self.activation_function_derivative(final_outputs)),
            numpy.transpose(neuron_outputs[len(neuron_outputs) - 1]))) * (1 + self.momentum)

        for i in range(len(self.hidden_layers) - 2, 0, -1):
            self.hidden_layers[i] += (self.learning_rate * numpy.dot(
                (neuron_errors[i] * self.activation_function_derivative(final_outputs)),
                numpy.transpose(neuron_outputs[i]))) * (1 + self.momentum)

        self.first_layer_weights += (self.learning_rate * numpy.dot(first_layer_errors *
                                                                    self.activation_function_derivative(neuron_outputs[0]),
                                                                    numpy.transpose(inputs)) * (1 + self.momentum))

    def query(self, input_list):
        # prepare array for neuron outputs
        neuron_outputs = []

        # convert inputs list to 2d array
        if self.bias:
            temp = copy.copy (input_list)
            temp.append (1)
            inputs = numpy.array (temp, ndmin=2).T
            # calculate signals into hidden layer
            neuron_outputs.append (numpy.dot (self.first_layer_weights, inputs))
        else:
            inputs = numpy.array(input_list, ndmin=2).T
            # calculate signals into hidden layer
            neuron_outputs.append (numpy.dot (self.first_layer_weights, inputs))

        for i in range (1, len (self.hidden_layers)):
            neuron_outputs.append (numpy.dot (self.hidden_layers[i], self.activation_function (neuron_outputs[i - 1])))

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function (numpy.dot (self.output_weights,
                                                             self.activation_function
                                                             (neuron_outputs[len (neuron_outputs) - 1])))

        return final_outputs
