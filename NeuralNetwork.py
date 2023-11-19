import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []  # List to store loss values for each iteration
        self.layers = []  # List to hold the architecture of the network
        self.data_layer = None  # Placeholder for input data and labels
        self.loss_layer = None  # Placeholder for the special layer providing loss and prediction

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return self.loss_layer.forward(input_tensor, label_tensor)

    def backward(self):
        label_tensor = self.loss_layer.label_tensor
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor