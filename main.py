import random

from Testing.Task2 import task_2
from mlp.NeuralNetwork import NeuralNetwork

#task_2()
network = NeuralNetwork(4, 2, 4, 0.1, 0.1, False, 2)
network.test(network.wih)

network.test(network.who)
