
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, momentum):
        self.input_nodes = inputnodes
        self.hidden_nodes = hiddennodes
        self.output_nodes = outputnodes
        # learning rate
        self.learning_rate = learningrate
        self.momentum = momentum
        pass
