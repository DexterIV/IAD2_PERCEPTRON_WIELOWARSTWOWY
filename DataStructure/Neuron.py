import random
import numpy


class Neuron:
    def __init__(self, number_of_inputs, values):
        self.values = values
        self.weights = numpy.zeros(number_of_inputs)

