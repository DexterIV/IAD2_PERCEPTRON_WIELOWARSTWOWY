import random
import decimal

import numpy

from DataStructure.NeuralNetwork import NeuralNetwork


'''
network = NeuralNetwork(4, 2, 4, 0.6, 0)

input_list = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
output = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
epochs = 10000

for e in range(3):
    network.train(input_list[e], output[e], 10000)
numpy.set_printoptions(suppress=True)  # avoid e-05 notation
fin = [network.query(input_list[0]),
       network.query(input_list[1]),
       network.query(input_list[2]),
       network.query(input_list[3])]
print(output)
print(fin)
'''