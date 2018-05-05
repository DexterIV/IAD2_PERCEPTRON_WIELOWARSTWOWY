from DataStructure.NeuralNetwork import NeuralNetwork

network = NeuralNetwork(4, 2, 4, 0.1, 0)

input = [0, 1, 0, 0]

output = [0, 1, 0, 0]

network.train(input, output, 1000)

fin = network.query(input)
print(input)
print(output)
print(fin)
