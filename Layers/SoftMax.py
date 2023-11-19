import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_probs = None

    def forward(self, input_data):
        # Compute SoftMax probabilities for each row in the input tensor
        exp_input_data = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        sum_exp_input = np.sum(exp_input_data, axis=1, keepdims=True)
        self.output_probs = exp_input_data / sum_exp_input
        return self.output_probs

    def backward(self, error_data):
        # Backward pass: calculate the gradient of the loss with respect to the SoftMax output
        loss_gradient = error_data - np.sum(error_data * self.output_probs, axis=1).reshape(-1, 1)
        gradient = loss_gradient * self.output_probs
        return gradient