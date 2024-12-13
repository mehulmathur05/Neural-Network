import numpy as np
from tensorflow.keras.datasets import mnist
import cv2
import pickle
import os
import time

# Utility functions for activation and loss
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def softmax(z):
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    # Clip probability to avoid log(0) and avoid unpredictable behaviour
    log_likelihood = -np.log(np.clip(y_pred[range(m), y_true], 1e-15, 1))
    loss = np.sum(log_likelihood) / m
    return loss

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.05, activation='relu'):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.decay_rate = 0.06
        self.activation = activation
        
        self.weights, self.biases = self.initialize_parameters()
    
    def initialize_parameters(self):
        np.random.seed(0)  # for reproducibility
        weights, biases = [], []
        for i in range(1, len(self.layer_sizes)):
            # Weights initialized with a small scale (0.01) to avoid large values at the beginning
            weights.append(np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * 0.01)
            biases.append(np.zeros((1, self.layer_sizes[i])))
        return weights, biases

    def activation_function(self, z):
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'sigmoid':
            return sigmoid(z)
        return z

    def activation_derivative(self, z):
        if self.activation == 'relu':
            return relu_derivative(z)
        elif self.activation == 'sigmoid':
            return sigmoid_derivative(z)
        return z

    def forward_propagation(self, X):
        Z_values, A_values = [None] * len(self.layer_sizes), [None] * len(self.layer_sizes)
        A_values[0] = X

        for i in range(1, len(self.layer_sizes) - 1):
            Z_values[i] = np.dot(A_values[i - 1], self.weights[i - 1]) + self.biases[i - 1]
            A_values[i] = self.activation_function(Z_values[i])
        
        # Output layer (Softmax)
        Z_values[-1] = np.dot(A_values[-2], self.weights[-1]) + self.biases[-1]
        A_values[-1] = softmax(Z_values[-1])

        return Z_values, A_values

    def backward_propagation(self, X, Y, Z_values, A_values):
        batch_size = X.shape[0]
        dW_values = [None] * (len(self.layer_sizes) - 1)
        db_values = [None] * (len(self.layer_sizes) - 1)

        # Output layer error values
        A_last = A_values[-1]
        dZ_last = A_last.copy()  # to avoid modifying A_last directly
        dZ_last[range(batch_size), Y] -= 1

        dW_values[-1] = np.dot(A_values[-2].T, dZ_last) / batch_size
        db_values[-1] = np.sum(dZ_last, axis=0, keepdims=True) / batch_size

        # Backpropagation through hidden layers
        for i in reversed(range(1, len(self.layer_sizes) - 1)):
            dA = np.dot(dZ_last, self.weights[i].T)
            dZ = dA * self.activation_derivative(Z_values[i])
            
            dW_values[i - 1] = np.dot(A_values[i - 1].T, dZ) / batch_size
            db_values[i - 1] = np.sum(dZ, axis=0, keepdims=True) / batch_size
            
            dZ_last = dZ

        return dW_values, db_values


    def update_parameters(self, dW_values, db_values):
        for i in range(len(self.layer_sizes) - 1):
            self.weights[i] -= self.learning_rate * dW_values[i]
            self.biases[i] -= self.learning_rate * db_values[i]
    
    def compute_loss(self, Y, A_last):
        return cross_entropy_loss(Y, A_last)
    
    def predict(self, X):
        _, A_values = self.forward_propagation(X)
        predictions = np.argmax(A_values[-1], axis=1)
        return predictions

    def train(self, X, Y, iterations, batch_size):
        m = X.shape[0]
        for iteration in range(iterations):

            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for j in range(0, m, batch_size):
                X_batch = X_shuffled[j : j+batch_size]
                Y_batch = Y_shuffled[j : j+batch_size]

                Z_values, A_values = self.forward_propagation(X_batch)
                loss = self.compute_loss(Y_batch, A_values[-1])
                dW_values, db_values = self.backward_propagation(X_batch, Y_batch, Z_values, A_values)
                self.update_parameters(dW_values, db_values)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.7f}")
                print("Current learning rate:", self.learning_rate)
                self.learning_rate = round(self.learning_rate*(1 - self.decay_rate), 4)

    def save_parameters(self):
        with open('weights.pkl', 'wb') as file: pickle.dump(self.weights, file)
        with open('biases.pkl', 'wb') as file:  pickle.dump(self.biases, file)
        print("Weights and biases saved to 'weights.pkl' and 'biases.pkl'.")

    def load_from_saved(self):
        with open('weights.pkl', 'rb') as file: self.weights = pickle.load(file)
        with open('biases.pkl', 'rb') as file:  self.biases = pickle.load(file)
        return

def save_misclassified(predictions, y_test):
    size = len(predictions)
    for i in range(size):
        if (predictions[i] != y_test[i] and not os.path.exists(f'img{i}.png')): 
            cv2.imwrite(f'img_{i}.png', np.resize(x_test[i], (28, 28))*255)
            print(f"index: {i}, correct value: {y_test[i]}, predicted value: {predictions[i]}")


if __name__ == '__main__':
    t_start = time.time()

    # Load and preprocess the dataset
    # x is the image pixel value data while y is the label
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalizing the data to a [0,1] range
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_test = x_test.reshape(-1, 28 * 28) / 255.0

    input_layer_size = x_train.shape[1]
    output_layer_size = 10  # 10 classes for MNIST
    hidden_layer_sizes = [256, 128]

    layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]

    nn = NeuralNetwork(layer_sizes, learning_rate=0.05, activation='relu')

    # Training the model
    iterations = 40
    batch_size = 64
    nn.train(x_train, y_train, iterations, batch_size)

    # nn.load_from_saved()

    # Testing
    predictions = nn.predict(x_test)
    accuracy = np.mean(predictions == y_test)
    print(f'Test accuracy: {accuracy * 100}%')

    t_end = time.time()

    time_elapsed = time.strftime("%M min %S sec", time.gmtime(t_end - t_start))
    print("Execution Time:", time_elapsed)

    # nn.save_parameters()
    # save_misclassified(predictions, y_test)
