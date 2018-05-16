import numpy
from library import funs
import copy

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, momentum, bias):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.bias = bias
        self.bias_set = True

        if self.bias:
            self.wih = (numpy.random.rand(self.hidden_nodes, self.input_nodes + 1) - 0.5)  # weight input to hidden
            self.wih = numpy.vstack([self.wih, numpy.zeros(self.input_nodes + 1)])
            self.who = (numpy.random.rand(self.output_nodes, self.hidden_nodes + 1) - 0.5)  # weight hidden to output
        else:
            self.wih = (numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)  # weight input to hidden
            self.who = (numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)  # weight hidden to output

        self.activation_function = lambda x: funs.sigmoid(x)
        self.activation_function_derivative = lambda x: funs.sigmoid_der(x)
        pass

    def train(self, input_list, target_list, epochs):
        for e in range(epochs):
            self.train_manual_epochs(input_list, target_list)
        pass

    def train_manual_epochs(self, input_list, target_list):
        # convert inputs list to 2d array

        if self.bias:
            temp = copy.copy(input_list)
            temp.append(1)
            inputs = numpy.array(temp, ndmin=2).T
            targets = numpy.array(target_list, ndmin=2).T
            # calculate signals into hidden layer
            hidden_inputs = numpy.dot(self.wih, inputs)
        else:
            inputs = numpy.array(input_list, ndmin=2).T
            targets = numpy.array(target_list, ndmin=2).T
            # calculate signals into hidden layer
            hidden_inputs = numpy.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # output layer error i)­ actual)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += (self.learning_rate * numpy.dot((output_errors * self.activation_function_derivative(final_outputs)),
                                                    numpy.transpose(hidden_outputs))) * (1 + self.momentum)

        self.wih += (self.learning_rate * numpy.dot((hidden_errors * self.activation_function_derivative(hidden_outputs)),
                                                    numpy.transpose(inputs))) * (1 + self.momentum)
        pass

    def query(self, input_list):
        if self.bias:
            temp = copy.copy(input_list)
            temp.append(1)
            inputs = numpy.array(temp, ndmin=2).T
            # calculate signals into hidden layer
            hidden_inputs = numpy.dot(self.wih, inputs)
        else:
            inputs = numpy.array(input_list, ndmin=2).T
            # calculate signals into hidden layer
            hidden_inputs = numpy.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)  # calculate the signals emerging from hidden layer

        final_outputs = numpy.dot(self.who, hidden_outputs)  # calculate signals into final output layer

        final_outputs = self.activation_function(
            final_outputs)  # calculate the signals emerging from final output layer

        return final_outputs
