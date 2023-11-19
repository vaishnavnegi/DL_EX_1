import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction = None
        self.label_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        epsilon = np.finfo(float).eps  # Small constant to prevent division by zero
        loss = -np.sum(np.log(prediction_tensor[label_tensor == 1] + epsilon))
        self.prediction = prediction_tensor
        self.label_tensor = label_tensor
        return loss

    def backward(self, label_tensor):
        # Backward pass: the error tensor is the gradient of the loss with respect to the prediction tensor
        epsilon = np.finfo(float).eps
        return -(label_tensor / (self.prediction + epsilon))
