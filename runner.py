import numpy as np
import NeuralNetwork as nn 

# Example usage
layers = (2, 3, 2)
data_lenght = 5
# Generate random input data
inputs = [np.random.standard_normal((layers[0], 1)) for _ in range(data_lenght)]

# Generate target data
targets = [[1, 0] if np.random.rand() > 0.5 else [0, 1] for _ in range(data_lenght)]
targets = [np.array(t).reshape(-1, 1) for t in targets]

# print(targets)
net = nn.NeuralNetwork(layers)
# loss = net.learn(inputs, targets,0.1)
# print("Initial Loss:", loss)

for i in range(500):
    loss = net.learn(inputs, targets,0.2)
    print("Initial Loss:", loss)

