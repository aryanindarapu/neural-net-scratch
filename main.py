import numpy as np
import pandas as pd
from neuron import DenseLayer
from NeuralNet import NeuralNet

data = pd.read_csv('./train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape

nn = NeuralNet(learningRate=0.01, totalIter=200)
print(nn.fit(X_train, Y_train))
nn.testPredict(0, X_dev, Y_dev)
nn.testPredict(1, X_dev, Y_dev)
nn.testPredict(2, X_dev, Y_dev)
nn.testPredict(3, X_dev, Y_dev)
nn.testPredict(4, X_dev, Y_dev)


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
