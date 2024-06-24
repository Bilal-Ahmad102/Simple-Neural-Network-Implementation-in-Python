import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.weight_shapes = [(a, b) for a, b in zip(layers[1:], layers[:-1])]
        self.weights = [np.random.standard_normal(s) for s in self.weight_shapes]
        self.biases = [np.zeros((s, 1)) for s in layers[1:]]
        self.gradient = []        
        self.gradientB = []
        self.correct_predicted = 0
    
    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.matmul(w, a) + b)
        return a

    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))
    
    def cost(self, output_activation, actual_value):
        error = output_activation - actual_value
        return np.sum(error ** 2)
    
    def loss_function(self, inputs, targets):
        total_cost = 0
        self.correct_predicted = 0  # Reset correct prediction count for this batch
        for x, y in zip(inputs, targets):
            output = self.predict(x)
            
            if output[0] > output[1]:
                predicted_label = [[1],[0]]
            else:
                predicted_label = [[0],[1]]
                
            if np.array_equal(predicted_label, y):
                self.correct_predicted += 1
            
            total_cost += self.cost(output, y)
        return total_cost / len(inputs)
    
    def learn(self, training_data, targets, learning_rate):
        h = 1e-5
        original_cost = self.loss_function(training_data, targets)
        
        self.gradient = []
        self.gradientB = []
        
        for l, (w_shape, w) in enumerate(zip(self.weight_shapes, self.weights)):
            layer_gradient = np.zeros_like(w)
            for i in range(w_shape[0]):
                for j in range(w_shape[1]):
                    self.weights[l][i, j] += h
                    delta_cost = self.loss_function(training_data, targets) - original_cost
                    self.weights[l][i, j] -= h
                    layer_gradient[i, j] = delta_cost / h
            self.gradient.append(layer_gradient)
          
        for l, b in enumerate(self.biases):
            layer_gradientB = np.zeros_like(b)
            for i in range(len(b)):
                self.biases[l][i] += h
                delta_cost = self.loss_function(training_data, targets) - original_cost
                self.biases[l][i] -= h
                layer_gradientB[i] = delta_cost / h
            self.gradientB.append(layer_gradientB)
        
        self.apply_gradient(learning_rate)        
        print(f"Correct Predicted: {self.correct_predicted}")
        return self.loss_function(training_data, targets)
       
    def apply_gradient(self, learning_rate):
        for l in range(len(self.weights)):
            self.weights[l] -= learning_rate * self.gradient[l]
        
        for l in range(len(self.biases)):
            self.biases[l] -= learning_rate * self.gradientB[l]
