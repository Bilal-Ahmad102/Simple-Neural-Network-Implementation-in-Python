**Getting Started**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Bilal-Ahmad102/Simple-Neural-Network-Implementation-in-Python 
   ```

2. **Install dependencies:**

   This code requires the `numpy` library for numerical computations. Install it using:

   ```bash
   pip install numpy
   ```

**Running the Example**

1. Navigate to the project directory:

   ```bash
   cd Neural_Network
   ```

2. Run the script:

   ```bash
   python runner.py
   ```

**Understanding the Code**

- **`NeuralNetwork` class:**
   - `__init__(self, layers)`: Initializes the network with the specified layer structure.
   - `predict(self, a)`: Performs forward propagation through the network, computing outputs for a given input.
   - `activation(x)`: Applies the sigmoid activation function (can be customized for different activation functions).
   - `cost(self, output_activation, actual_value)`: Calculates the mean squared error between the network's output and the target value.
   - `loss_function(self, inputs, targets)`: Calculates the total loss for a batch of training data, also keeping track of correct predictions.
   - `learn(self, training_data, targets, learning_rate)`: Trains the network using gradient descent based on the provided training data, targets, and learning rate.
   - `apply_gradient(self, learning_rate)`: Updates the weights and biases using the calculated gradients.

- **Example usage:**
   - Defines layers, data length, and generates random training data (inputs and targets).
   - Creates a `NeuralNetwork` instance.
   - Trains the network for a specified number of epochs, printing the initial loss on each iteration.

**Customization**

The provided code serves as a foundation for exploration. You can customize it by:

- Trying different network architectures (number and size of layers)
- Implementing different activation functions (e.g., ReLU, tanh)
- Using other loss functions (e.g., cross-entropy for classification problems)
- Incorporating regularization techniques (e.g., L1/L2 regularization)
- Experimenting with different learning rate schedules

**Further Exploration**

For a more comprehensive understanding of neural networks, consider exploring:

- Deep learning frameworks like TensorFlow, PyTorch, or Keras for advanced capabilities.
- Public datasets to train on more complex tasks.
- Online resources and tutorials on neural network architectures, hyperparameter tuning, and advanced training techniques.

This repository provides a stepping stone for your journey into the exciting world of neural networks!