import numpy as np
import pandas as pd
from NeuralNet import NeuralNet

imageData = pd.read_csv('./train.csv')
imageData = np.array(imageData)
np.random.shuffle(imageData)

imageData_train = imageData[1000:imageData.shape[0]].T
xTrain = imageData_train[1:imageData.shape[1]] / 255.0
yTrain = imageData_train[0]

imageData_test = imageData[0:1000].T
xTest = imageData_test[1:imageData.shape[1]] / 255.0
yTest = imageData_test[0]

network = NeuralNet(learningRate=0.01, totalIter=200)
network.fit(xTrain, yTrain)
network.testPredict(0, xTest, yTest)
network.testPredict(1, xTest, yTest)
network.testPredict(2, xTest, yTest)
network.testPredict(3, xTest, yTest)
network.testPredict(4, xTest, yTest)

network.predict(xTest, yTest)
print("Output", network.output)


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
