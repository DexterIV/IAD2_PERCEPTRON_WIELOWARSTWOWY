import random
import numpy


class Perceptron:
    def __init__(self, number_of_inputs, values, random_seed):
        self.values = values
        random.seed(random_seed)
        self.weights = [random.random() for _ in range(number_of_inputs)]

