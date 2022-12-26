import numpy as np

np.random.seed(0)

def dot_product(list1, list2):
    if len(list1) != len(list2): return
    output = 0
    for l1, l2 in zip(list1, list2): output += l1 * l2
    return output 

class DenseLayer:
    def __init__(self, numInputs, numNeurons, inputs):
        self.inputs = inputs
        self.weights = 0.10 * np.random.randn(numInputs, numNeurons) # removes need for transpose
        self.biases = np.zeros((1, numNeurons))
        
    def forwardProp(self):
        self.output = np.dot(self.inputs, self.weights) + self.biases
        
    def activationReLU(self):
        self.output = np.maximum(0, self.inputs)
        
    def activationSoftmax(self):
        # write softmax function here
        pass
        