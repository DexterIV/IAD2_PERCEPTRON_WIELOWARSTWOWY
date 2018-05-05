import numpy
import library


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, momentum):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.wih = (numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)  # weight input to hidden
        self.who = (numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)  # weight hidden to output
        self.activation_function = lambda x: library.sigmoid(x)

    def train(self):
        return

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T # convert inputs list to 2d array

        hidden_inputs = numpy.dot(self.wih, inputs) # calculate signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # calculate the signals emerging from hidden layer

        final_outputs = numpy.dot(self.who, hidden_outputs) # calculate signals into final output layer

        final_outputs = self.activation_function(final_outputs) # calculate the signals emerging from final output layer


        return final_outputs
