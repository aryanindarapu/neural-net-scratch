import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, layers=[10, 784, 1], learningRate=0.001, totalIter=1000):
        self.input = None
        self.output = None
        self.givenOutput = None
        self.learningRate = learningRate
        self.loss = []
        self.layers = layers
        self.totalIter = totalIter
        
    def initWeights(self, randSeed = None, W1 = None, b1 = None, W2 = None, b2 = None):
        # Initialize Weights
        if randSeed == None:
            self.W1 = W1
            self.b1 = b1
            self.W2 = W2
            self.b2 = b2
        else:
            np.random.seed(randSeed)
            self.W1 = np.random.rand(self.layers[0], self.layers[1]) - 0.5
            self.b1 = np.random.rand(self.layers[0], self.layers[2]) - 0.5
            self.W2 = np.random.rand(self.layers[0], self.layers[0]) - 0.5
            self.b2 = np.random.rand(self.layers[0], self.layers[2]) - 0.5
        
    def getWeights(self):
        return self.W1, self.b1, self.W2, self.b2
        
    def relu(self, X):
        return np.maximum(X, 0)
    
    def dRelu(self, X):
        X[X <= 0] = 0
        X[X > 0] = 1
        return X    
        
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    # def dSigmoid()
    
    def softmax(self, X):
        A = np.exp(X) / sum(np.exp(X))
        return A
    
    # def rmsLoss(self, output, givenOutput):
    #     result = output - givenOutput
    #     return np.sqrt(result.T )
    
    def crossentropyLoss(self, output, givenOutput):
        # works around np.log(0) errors
        output = np.maximum(output, 0.0000000001)
        givenOutput = np.maximum(givenOutput, 0.0000000001)
        
        return -1 / len(givenOutput) * (np.sum(np.multiply(np.log(output), givenOutput) + np.multiply((1.0 - givenOutput), np.log(1.0 - output))))

    def forwardProp(self):
        Z1 = np.dot(self.W1, self.input) + self.b1
        A1 = self.relu(Z1)
        # print(A1)
        Z2 = np.dot(self.W2, A1) + self.b2
        self.output = self.softmax(Z2)
        # self.output = self.sigmoid(Z2)
        
        return Z1, A1, Z2 # , self.crossentropyLoss(self.output, self.givenOutput)
        
        
    def backwardProp(self, Z1, A1, Z2):
        # '''
        one_hot_Y = np.zeros((self.givenOutput.size, self.givenOutput.max() + 1))
        one_hot_Y[np.arange(self.givenOutput.size), self.givenOutput] = 1
        one_hot_Y = one_hot_Y.T
        
        dZ2 = self.output - one_hot_Y
        dW2 = (1 / len(self.input)) * np.dot(dZ2, A1.T)
        db2 = (1 / len(self.input)) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(self.W2.T, dZ2) * self.dRelu(Z1)
        dW1 = (1 / len(self.input)) * np.dot(dZ1, self.input.T)
        db1 = (1 / len(self.input)) * np.sum(dZ1, axis=1, keepdims=True)
        # '''
        
        # Updating weights and biases
        self.W1 = self.W1 - self.learningRate * dW1
        self.W2 = self.W2 - self.learningRate * dW2
        self.b1 = self.b1 - self.learningRate * db1
        self.b2 = self.b2 - self.learningRate * db2
        
    def getAccuracy(self, output):
        print(output, self.givenOutput)
        return np.sum(output == self.givenOutput) / self.givenOutput.size
        
    def fit(self, input, givenOutput, randSeed=14351):
        self.input = input
        self.givenOutput = givenOutput
        self.initWeights(randSeed)
        
        for i in range(self.totalIter):
            Z1, A1, Z2 = self.forwardProp()
            self.backwardProp(Z1, A1, Z2)
            # self.loss.append(loss)
            if i % 10 == 0:
                print("Iteration: ", i)
                pred = np.argmax(self.output, 0)
                print(self.getAccuracy(pred))
            
        return self.output
        
    def predict(self, input, output):
        self.input = input
        self.givenOutput = output
        self.forwardProp()
        self.output = np.argmax(self.output, 0)

    def testPredict(self, idx, input, output):
        self.input = input[:, idx, None]        
        currOut = input[:, idx, None]
        self.forwardProp()
        print("Prediction: ", np.argmax(self.output, 0))
        print("Label: ", output[idx])
        
        currOut = currOut.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(currOut, interpolation='nearest')
        plt.show()
        