import numpy as np

class Dense:
    
    def __init__(self, n_inputs, n_neurons):
        """ Initialize the weights and biases of each neurons
        n_inputs = number of input features
        n_neurons = number of desired neurons
        """
        # using np.random.randn and * 0.01 is to break the symetry of the neurons
        
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        # biases can be initialize as zeros
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        """ Calculate the output layer using The Dot product of input feature and weight plus bias
        Input:
        inputs = Training examples
        
        Output:
        output = Output of the training example
        """
        # calculate the output layer
        self.inputs = inputs
        output = np.dot(inputs, self.weights) + self.biases
        
        return output
    
    def backward(self, dvalues):
        """Calculate gradient descent on parameter

        Args:
            dvalues ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.weights = np.dot(self.inputs.T, dvalues)
        self.biases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
    
# ReLU activation
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        output = np.maximum(0, inputs)
        
        return output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        
        self.dinputs[self.inputs <=0] = 0
    
# Sotfmax activation
class Activation_Softmax:
    def forward(self, inputs):
        # input - np.max to prevent the exponential function from overflowing
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        softmax = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return softmax
   
    
# common loss class
class Loss:

    def calc(self, output, y):
        """ Calculate the data and regularization losses
        Input: 
            output : The output of the activation function
            y : Class targets
            
        Output:
            data_loss : Average Loss
        """
        data_loss = self.forward(output, y)
        
        return data_loss
    

# categorical cross entropy loss 
class Loss_CategoricalCrossentropy(Loss):
    
    def forward(self, y_pred, y):
        """Calculate Cross-entropy loss
        Input : 
            y_pred : The output of the activation function
            y : Class targets
            
        Output :
            average_loss : Average Loss
        """
        # len of training example
        m = len(y_pred)
        
        # clip data to prevent divison by 0
        # clip both sides to not drag mean toward any values
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        # probabilities for target values only if categorical labels
        if len(y.shape) == 1:
            correct_conf = y_pred_clipped[range(len(y_pred_clipped)), y]
        # one hot encoded
        elif len(y.shape) == 2:
            correct_conf = np.sum(y_pred_clipped * y, axis=1, keepdims=True)

        # compute loss
        average_loss = 1/m * np.sum(-np.log(y_pred_clipped[range(len(y_pred_clipped)), y]))
        
        return average_loss