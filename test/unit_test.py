import os
import sys
import unittest
import numpy as np

# 使用相对路径：从当前文件向上找项目根目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_file_dir, '..')
sys.path.insert(0, project_root)

import networklibs as netlib

class TestsNNs(unittest.TestCase):
    def test_Neuron(self): # test Neuron unit output
        weights = [[0, 1]]
        inputs = [[2], [3]]
        bias = [[4]]
        
        desired_output = netlib.network_units.sigmoid().forward([[7]])
        nn_test = netlib.network_units.Neuron(weights, bias, netlib.network_units.sigmoid)
        
        self.assertEqual(nn_test.forward(inputs), desired_output)
        
    def test_OneHiddenNNs(self): # test NNs with 1 hidden layer (2 units). layer shape (2, 2) and (2, 1)
        weights_1 = [[0, 1], [0, 1]]
        bias_1 = [[0], [0]]
        weights_2 = [[0, 1]]
        bias_2 = [[0]]
        test_activation_func = netlib.network_units.sigmoid()
        
        inputs = [[2], [3]]
        
        desired_output_1 = test_activation_func.forward([[3], [3]])
        desired_output_2 = test_activation_func.forward(np.matmul(weights_2, desired_output_1)+bias_2)
        
        nn_test_1 = netlib.network_units.Neuron(weights_1, bias_1, test_activation_func)
        nn_test_2 = netlib.network_units.Neuron(weights_2, bias_2, test_activation_func)
        
        outputs_1 = nn_test_1.forward(inputs)
        outputs_2 = nn_test_2.forward(outputs_1)
        
        print(nn_test_1.get_activation_gradient())
        print(nn_test_2.get_activation_gradient())
        
        self.assertTrue(np.array_equal(outputs_1, desired_output_1))
        self.assertTrue(np.array_equal(outputs_2, desired_output_2))
        
    def test_NoHiddenNNs(self): # test NNs with 0 hidden layer. layer shape (2, 1)
        weights_1 = [[0, 1], [0, 1]]
        bias_1 = [[0], [0]]
        test_activation_func = netlib.network_units.sigmoid()
        
        inputs = [[-2], [-1]]
        
        desired_output_1 = test_activation_func.forward([[-1], [-1]])
        
        nn_test_1 = netlib.network_units.Neuron(weights_1, bias_1, test_activation_func)
        
        outputs_1 = nn_test_1.forward(inputs)
        
        self.assertTrue(np.array_equal(outputs_1, desired_output_1))
        
        
if __name__ == '__main__':
    unittest.main()