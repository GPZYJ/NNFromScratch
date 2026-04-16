import numpy as np
import os
import sys
import matplotlib.pyplot as plt


current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_file_dir, '..')
sys.path.insert(0, project_root)

import networklibs as netlib

def mse_loss(y_true, y_pred):
    
    return ((y_pred - y_true)**2).mean()


class TesNN:
    '''
    A neuron network with:
        - 1 input
        - 1 hidden layer with 2 neurons
        - 1 output 
    '''
    
    def __init__(self, NN_shape, use_bias=True, init_method=None, activation_func=None): 
        '''
            NN_shap the shape of NNs including input shape, for example:1 * 2 * 1
            init_method: np.random.normal(mean value, standard deviation, (output_shape, input_shape))、 np.random.random((output_shape, input_shape))
            
        '''
        
        self.NN_shape = NN_shape
        self.use_bias = use_bias
        
        self.NNs = []
        
        for i in range(len(NN_shape)-1):
            
            input_shape = NN_shape[i]
            output_shape = NN_shape[i+1]
            
            if self.use_bias:
                weights = np.random.random((output_shape, input_shape))
                bias = np.random.random((output_shape, 1))
                nn = netlib.network_units.Neuron(weights, bias, activation_func=activation_func)
                self.NNs.append(nn)
            else:
                weights = np.random.random((output_shape, input_shape))
                nn = netlib.network_units.Neuron(weights, activation_func=activation_func)
                self.NNs.append(nn)
        
    def forward(self, x):
        
        output = x.copy()
        for nn in self.NNs:
            output = nn.forward(output)
        
        return output
    
    def train(self, data_x, data_y, epochs, learning_rate):
        losses = []
        loss_epoch = []
        for epoch in range(epochs):
                
            y_pred = self.forward(data_x)
            d_L_d_ypred = -2 * (data_y - y_pred)
            
            gradient = d_L_d_ypred
            
            layers_num = len(self.NNs)
            old_weights = None
            if self.use_bias:
                for i in range(layers_num):
                    if i == 0: 
                        # weights update gradient is (derive of activation) *(matrix product) (derive of Loss function) * (layer input)^T
                        # bias update gradient is (derive of activation) *(matrix product) (derive of Loss function)
                        
                        activation_gradient = self.NNs[layers_num-i-1].get_activation_gradient()
                        gradient = np.matmul(gradient, activation_gradient)
                        
                        old_weights = self.NNs[layers_num-i-1].weights.copy()
                        old_bias = self.NNs[layers_num-i-1].bias.copy()
                        
                        new_weights = old_weights - learning_rate * np.mean(np.matmul(self.NNs[layers_num-i-1].inputs, gradient), axis=0).T
                        new_bias = old_bias - learning_rate * np.mean(gradient, axis=0).T
                        
                        self.NNs[layers_num-i-1].update(new_weights, new_bias)
                        
                    else: # update gradient is (derive of activation) * (previous layer weight)^T * (layer input)^T
                        
                        activation_gradient = self.NNs[layers_num-i-1].get_activation_gradient()
                        gradient = np.matmul(gradient, old_weights)
                        gradient = np.matmul(gradient, activation_gradient)
                        
                        old_weights = self.NNs[layers_num-i-1].weights.copy()
                        old_bias = self.NNs[layers_num-i-1].bias.copy()
                        
                        new_weights = old_weights - learning_rate * np.mean(np.matmul(self.NNs[layers_num-i-1].inputs, gradient), axis=0).T
                        new_bias = old_bias - learning_rate * np.mean(gradient, axis=0).T
                        
                        self.NNs[layers_num-i-1].update(new_weights, new_bias)
                        
            else:
                for i in range(layers_num):
                    if i == 0: 
                        # weights update gradient is (derive of activation) *(matrix product) (derive of Loss function) * (layer input)^T
                        # bias update gradient is (derive of activation) *(matrix product) (derive of Loss function) * (layer input)^T
                        
                        activation_gradient = self.NNs[layers_num-i-1].get_activation_gradient()
                        gradient = np.matmul(gradient, activation_gradient)
                        
                        old_weights = self.NNs[layers_num-i-1].weights.copy()
                        old_bias = self.NNs[layers_num-i-1].bias.copy()
                        
                        new_weights = old_weights - learning_rate * np.mean(np.matmul(self.NNs[layers_num-i-1].inputs, gradient), axis=0).T
                        
                        self.NNs[layers_num-i-1].update(new_weights)
                        
                    else: # update gradient is (derive of activation) * (previous layer weight)^T * (layer input)^T
                        
                        activation_gradient = self.NNs[layers_num-i-1].get_activation_gradient()
                        gradient = np.matmul(gradient, old_weights)
                        gradient = np.matmul(gradient, activation_gradient)
                        
                        old_weights = self.NNs[layers_num-i-1].weights.copy()
                        old_bias = self.NNs[layers_num-i-1].bias.copy()
                        
                        new_weights = old_weights - learning_rate * np.mean(np.matmul(self.NNs[layers_num-i-1].inputs, gradient), axis=0).T
                        
                        self.NNs[layers_num-i-1].update(new_weights)
                            
                            
            if epoch % 10 == 0:
                
                y_preds = np.array([self.forward(sample) for sample in data_x])
                
                loss = mse_loss(data_y, y_preds)
                losses.append(loss)
                loss_epoch.append(epoch)
                print("Epoch %d loss: %.3f" % (epoch, loss))
                    
        # plot loss curve
        plt.figure(figsize=(10, 6))

        plt.plot(loss_epoch, losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
if __name__ == '__main__':
    
    np.random.seed(123)
    
    data = np.array([
    [-2, -1],  # Alice
    [25, 6],   # Bob
    [17, 4],   # Charlie
    [-15, -6], # Diana
    ])
    
    all_y_trues = np.array([
    1, # Alice
    0, # Bob
    0, # Charlie
    1, # Diana
    ])
    
    NN_shape = [2, 2, 1, 1]
    activation_func=netlib.network_units.sigmoid()
    NN = TesNN(NN_shape, activation_func=activation_func)

    NN.train(data.reshape(data.shape[0], -1, 1), all_y_trues.reshape(-1, 1, 1), 10000, 0.1)
