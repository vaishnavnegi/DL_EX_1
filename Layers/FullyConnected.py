from .Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (self.input_size + 1, self.output_size))
        self.input_tensor = None
        self.gradient_weights = None
        self.optimizer = None

    # def forward(self, input_tensor):
    #     self.input_tensor = input_tensor
    #     input_tensor = np.array([np.append(row, 1) for row in input_tensor])
    #     return np.dot(input_tensor, self.weights)

    def forward(self , input_tensor):
        biases = np.ones(input_tensor.shape[0], dtype=int).reshape(input_tensor.shape[0] , 1)
        self.input_tensor = np.concatenate((input_tensor , biases) , axis = 1)
        return np.dot(self.input_tensor , self.weights)


    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return np.dot(error_tensor , self.weights.T)[:,:-1]

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value





