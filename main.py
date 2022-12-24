import numpy as np 
from neuron import DenseLayer

inputs = np.array([[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])

layer1 = DenseLayer(4, 5)
layer2 = DenseLayer(5, 3)

layer1.forwardProp(inputs)
layer2.forwardProp(layer1.output)
print(layer2.output)

''' Old Single Layer Code
weights = np.array([[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]])
biases = [2, 3, 0.5]

print(np.dot(inputs, weights.T) + biases)

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for neuron_input, weight in zip(inputs, neuron_weights):
        neuron_output += neuron_input * weight

    
    layer_outputs.append(neuron_output + neuron_bias)
    
print(layer_outputs)
'''
