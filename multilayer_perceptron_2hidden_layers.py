#
import torch
import torch.nn.functional as F
import numpy as np

#
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 3rd layer
            torch.nn.Linear(20, 10),
            torch.nn.ReLU(),

            # 4th hidden layer
            torch.nn.Linear(10, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities
#
model = NeuralNetwork(40 , 5)

#
inputs_numpy = 1 + 19 * np.random.rand(1, 40)

#
input_tensor = torch.tensor(inputs_numpy).to(torch.float32)

#
result = model.forward(input_tensor)

#
print(f"The output of the model is: {result}")

#
print(f"The weights at the first layer: {model.layers[0].weight}")
print(f"The bias at the first layer: {model.layers[0].bias}")
print(f"The weights at the second layer: {model.layers[2].weight}")
print(f"The bias at the second layer: {model.layers[2].bias}")
print(f"The weights at the third layer: {model.layers[4].weight}")
print(f"The bias at the third layer: {model.layers[4].bias}")