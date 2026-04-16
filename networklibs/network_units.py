import numpy as np


class sigmoid:
    @staticmethod
    def forward(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        f = 1 / (1 + np.exp(-x))
        return f
    @staticmethod
    def derivative(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        f = 1 / (1 + np.exp(-x))
        return f * (1-f)

class tanh:
    @staticmethod
    def forward(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        f = (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))
        return f
    @staticmethod
    def derivative(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        f = (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))
        return 1 - f**2


''' Neuron units of NNs '''
class Neuron:
    def __init__(self, weights, bias=None, activation_func=None):  # weights: (N_out, N_in), bias: (N_out)
        self.weights = np.array(weights)
        if bias is not None:
            self.bias = np.array(bias)
        else:
            self.bias = None
        self.activation_func = activation_func
        self.pre_activation_value = None
        self.inputs = None
        
    def update(self, weights, bias=None): # update when back-propagation
        self.weights = np.array(weights)
        if self.bias is not None:
            self.bias = np.array(bias)
    
    def forward(self, inputs): # calculate the output using Matrix Product
        inputs = np.array(inputs)
        self.inputs = inputs.copy()
        if self.bias is not None:
            outputs = np.matmul(self.weights, inputs) + self.bias
        else:
            outputs = np.matmul(self.weights, inputs)
        # ToDO add activation function
        self.pre_activation_value = outputs.copy()
        if self.activation_func:
            outputs = self.activation_func.forward(outputs)
        return outputs
    
    def get_activation_gradient(self):
        activation_gradient = self.activation_func.derivative(self.pre_activation_value)
        
        def numpy_to_diag_loop(x):
            batch_size, n, _ = x.shape
            result = np.zeros((batch_size, n, n))
            
            for i in range(batch_size):
                result[i] = np.diag(x[i].flatten())
            
            return result
        
        if len(activation_gradient.shape) == 3:
            layer_activation_gradient = numpy_to_diag_loop(activation_gradient)
        else:
            
            layer_activation_gradient = np.diag(activation_gradient.flatten())
        
        return layer_activation_gradient
        
