from .Base import BaseLayer
import numpy as np
class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # Apply the ReLU activation function element-wise
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        # Backward pass: multiply the error tensor by 1 where the input was greater than 0
        return error_tensor * (self.input_tensor > 0)