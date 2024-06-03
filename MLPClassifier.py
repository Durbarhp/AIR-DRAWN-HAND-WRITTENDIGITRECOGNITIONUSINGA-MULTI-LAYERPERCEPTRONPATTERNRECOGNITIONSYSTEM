#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pickle

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parameters = self.initialize_parameters()

    # Activation function: Rectified Linear Unit (ReLU)
    def relu(self, x):
        """
        Compute the Rectified Linear Unit (ReLU) activation.

        Parameters:

        - x: Input data

        Returns:
        - Activated output
        """
        return np.maximum(0, x)

    # Derivative of ReLU for backpropagation
    def relu_derivative(self, x):
        """
        Compute the derivative of the Rectified Linear Unit (ReLU) activation.

        Parameters:
        - x: Input data

        Returns:
        - Derivative of the activation
        """
        return np.where(x > 0, 1, 0)

    # Activation function: Softmax
    def softmax(self, x):
        """
        Compute the softmax activation for a set of raw scores.

        Parameters:
        - x: Input data (raw scores)

        Returns:
        - Probabilities after softmax activation
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Initialize weights and biases for the neural network
    def initialize_parameters(self):
        """
        Initialize the weights and biases for the neural network.

        Parameters:
        - input_size: Number of features in the input data
        - hidden_size: Number of neurons in the hidden layer
        - output_size: Number of neurons in the output layer

        Returns:
        - Dictionary containing initialized weights and biases
        """
        np.random.seed(42)  # Set a seed for reproducibility
        weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01  # Initialize weights for input to hidden layer with small random values
        biases_hidden = np.zeros((1, self.hidden_size))  # Initialize biases for the hidden layer with zeros
        weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01  # Initialize weights for hidden to output layer with small random values
        biases_output = np.zeros((1, self.output_size))  # Initialize biases for the output layer with zeros
        # Return a dictionary containing initialized weights and biases
        return {
            'W1': weights_input_hidden,
            'b1': biases_hidden,
            'W2': weights_hidden_output,
            'b2': biases_output
        }

    # Forward pass through the neural network
    def forward_pass(self, X):
        """
        Perform the forward pass through the neural network.

        Parameters:
        - X: Input data
        - parameters: Dictionary containing weights and biases

        Returns:
        - Dictionary containing intermediate and final layer outputs
        """
        # Input to hidden layer
        Z1 = np.dot(X, self.parameters['W1']) + self.parameters['b1'] # Compute the weighted sum of inputs for the hidden layer
        A1 = self.relu(Z1) # Apply the Rectified Linear Unit (ReLU) activation to the hidden layer outputs

        # Hidden to output layer
        Z2 = np.dot(A1, self.parameters['W2']) + self.parameters['b2'] # Compute the weighted sum of inputs for the output layer
        A2 = self.softmax(Z2) # Apply the softmax activation to the output layer outputs

        # Return a dictionary containing intermediate and final layer outputs
        return {
            'Z1': Z1, # Weighted sum of inputs for the hidden layer
            'A1': A1, # Output of the hidden layer after ReLU activation
            'Z2': Z2, # Weighted sum of inputs for the output layer
            'A2': A2  # Output of the output layer after softmax activation
        }


    # Compute the cross-entropy loss
    def compute_loss(self, predictions, y):
        """
        Compute the cross-entropy loss given predicted probabilities and true labels.

        Parameters:
        - predictions: Predicted probabilities
        - y: True labels

        Returns:
        - Cross-entropy loss
        """
        m = y.shape[0] # Get the number of examples in the batch
        log_probs = -np.log(predictions[range(m), y]) # Calculate negative log probabilities for the true labels
        loss = np.sum(log_probs) / m # Compute the average cross-entropy loss over the batch
        return loss


    # Backward pass to update weights and biases using gradient descent
    def backward_pass(self, X, y, cache, learning_rate):
        """
        Perform the backward pass to update the weights and biases using gradient descent.

        Parameters:
        - X: Input data
        - y: True labels
        - parameters: Dictionary containing weights and biases
        - cache: Dictionary containing intermediate layer outputs
        - learning_rate: Learning rate for gradient descent
        """
        m = X.shape[0]  # Number of examples in the batch
        # Compute the derivative of the loss with respect to the output layer activations
        dZ2 = cache['A2'] - np.eye(self.output_size)[y]
        # Compute gradients for the output layer weights and biases
        dW2 = np.dot(cache['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        # Compute the derivative of the loss with respect to the hidden layer activations
        dA1 = np.dot(dZ2, self.parameters['W2'].T)
        # Compute the derivative of the loss with respect to the hidden layer weighted sum
        dZ1 = dA1 * self.relu_derivative(cache['Z1'])
        # Compute gradients for the hidden layer weights and biases
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update parameters using gradient descent
        self.parameters['W1'] -= learning_rate * dW1
        self.parameters['b1'] -= learning_rate * db1
        self.parameters['W2'] -= learning_rate * dW2
        self.parameters['b2'] -= learning_rate * db2




    # Training the neural network
    def train(self, X_train, y_train, learning_rate, epochs):
        """
        Train the neural network using the provided training data and labels.

        Parameters:
        - X_train: Training data
        - y_train: Training labels
        - hidden_size: Number of neurons in the hidden layer
        - output_size: Number of neurons in the output layer
        - learning_rate: Learning rate for gradient descent
        - epochs: Number of training epochs
        Returns:
        - Trained parameters
        """
        losses = [] # List to store the loss at each epoch

        for epoch in range(epochs):
            # Forward pass through the neural network
            cache = self.forward_pass(X_train)

            # Compute the cross-entropy loss
            loss = self.compute_loss(cache['A2'], y_train)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

            # Backward pass to update weights and biases using gradient descent
            self.backward_pass(X_train, y_train, cache, learning_rate)

        # Plot the loss curve
        plt.plot(losses)
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    # Predict using the trained model
    def predict(self, X):
        """
        Make predictions using the trained model.
        Parameters:
        - X: Input data
        - parameters: Trained parameters
        Returns:
        - Predicted labels
        """
        # Forward pass through the neural network to get predicted probabilities
        cache = self.forward_pass(X)
        return np.argmax(cache['A2'], axis=1) # Return the predicted labels as the indices of the maximum probabilities

    # Save Model
    def save_model(self, filename):
        """
        Save the trained neural network model parameters to a file using Pickle.

        Parameters:
        - filename (str): The name of the file to save the model to.

        Returns:
        None
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.parameters, file)
        print(f'Model saved as {filename}')

    # Load Model
    def load_model(self, filename):
        """
        Load a previously saved model from a file.

        Parameters:
        - filename (str): The path to the file containing the saved model.

        Returns:
        None
        """
        with open(filename, 'rb') as file:
            self.parameters = pickle.load(file)
        # print(f'Model loaded from {filename}')

