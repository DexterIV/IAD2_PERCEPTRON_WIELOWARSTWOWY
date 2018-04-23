import random
import numpy


class Neuron:
    def __init__(self, number_of_inputs):
        self.value = random.Random(1)
        self.weights = numpy.zeros(number_of_inputs)
