import torch
import torch.nn as nn
import random


class neuroNet(nn.Module):
    """
    Neural Network implementation for snake game decision making.

    Implements a feedforward neural network with two hidden layers using PyTorch.
    The network takes environmental state as input and outputs action probabilities
    for four possible movement directions. Uses ReLU activation for hidden layers
    and sigmoid for output layer.

    Attributes:
        a (int): Size of input layer (environmental state features)
        b (int): Size of first hidden layer
        c (int): Size of second hidden layer
        d (int): Size of output layer (action probabilities)
        fc1 (nn.Linear): First fully connected layer (input -> hidden1)
        fc2 (nn.Linear): Second fully connected layer (hidden1 -> hidden2)
        out (nn.Linear): Output layer (hidden2 -> output)
        relu (nn.ReLU): ReLU activation function for hidden layers
        sigmoid (nn.Sigmoid): Sigmoid activation function for output layer
    """

    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, weights):
        """
        Initialize neural network with specified architecture and weights.

        Args:
            n_input (int): Number of input features (environmental state size)
            n_hidden1 (int): Number of neurons in first hidden layer
            n_hidden2 (int): Number of neurons in second hidden layer
            n_output (int): Number of output neurons (number of possible actions)
            weights (list): Flattened weight parameters for entire network
        """
        super(neuroNet, self).__init__()

        # Store layer dimensions for weight management
        self.a = n_input
        self.b = n_hidden1
        self.c = n_hidden2
        self.d = n_output

        # Define network architecture with linear layers
        self.fc1 = nn.Linear(n_input, n_hidden1)  # Input to first hidden layer
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)  # First to second hidden layer
        self.out = nn.Linear(n_hidden2, n_output)  # Second hidden to output layer

        # Define activation functions
        self.relu = nn.ReLU()  # Non-linear activation for hidden layers
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for output probabilities

        # Initialize network with provided weights
        self.update_weights(weights)

    def forward(self, x):
        """
        Forward pass through the neural network.

        Processes input through all layers with appropriate activations to
        generate action probabilities for snake movement decision.

        Args:
            x (torch.Tensor): Input tensor containing environmental state

        Returns:
            torch.Tensor: Output probabilities for each possible action
        """
        # First hidden layer with ReLU activation
        y = self.fc1(x)
        y = self.relu(y)

        # Second hidden layer with ReLU activation
        y = self.fc2(y)
        y = self.relu(y)

        # Output layer with sigmoid activation (probability values 0-1)
        y = self.out(y)
        y = self.sigmoid(y)

        return y

    def update_weights(self, weights):
        """
        Update neural network weights from flattened weight array.

        Takes a flattened array of weights and biases and properly assigns them
        to the corresponding network layers. Weight array must contain exactly
        the right number of parameters for the network architecture.

        Args:
            weights (list): Flattened array containing all network weights and biases
                          Format: [fc1_weights, fc1_biases, fc2_weights, fc2_biases,
                                  out_weights, out_biases]
                          Total length: a*b+b + b*c+c + c*d+d
        """
        weights = torch.FloatTensor(weights)

        # Disable gradient computation for weight assignment
        with torch.no_grad():
            # Calculate array indices for each layer's parameters
            x = self.a * self.b  # End of fc1 weights
            xx = x + self.b  # End of fc1 biases
            y = xx + self.b * self.c  # End of fc2 weights
            yy = y + self.c  # End of fc2 biases
            z = yy + self.c * self.d  # End of output weights
            # Remaining elements are output biases

            # Assign weights and biases to first fully connected layer
            self.fc1.weight.data = weights[0:x].reshape(self.b, self.a)
            self.fc1.bias.data = weights[x:xx]

            # Assign weights and biases to second fully connected layer
            self.fc2.weight.data = weights[xx:y].reshape(self.c, self.b)
            self.fc2.bias.data = weights[y:yy]

            # Assign weights and biases to output layer
            self.out.weight.data = weights[yy:z].reshape(self.d, self.c)
            self.out.bias.data = weights[z:]

    def predict_next_action(self, input):
        """
        Generate action prediction from environmental state input.

        Takes current game state, processes it through the neural network,
        and returns the index of the action with highest probability.
        Used for actual game decision making.

        Args:
            input (list): Environmental state features as list of floats

        Returns:
            int: Index of predicted best action (0-3 for four directions)
        """
        # Convert input to tensor format expected by PyTorch
        input = torch.tensor([input]).float()

        # Forward pass through network to get action probabilities
        y = self(input)

        # Return index of action with highest probability
        return torch.argmax(y, dim=1).tolist()[0]


if __name__ == '__main__':
    # Example usage and testing of neural network
    # Generate random weights for testing network functionality
    weights = [random.random() for i in range(32 * 20 + 20 * 12 + 12 * 4 + 20 + 12 + 4)]

    # Create network instance with test weights
    model = neuroNet(32, 20, 12, 4, weights)

    # Generate random input for testing
    input = [random.random() for _ in range(32)]

    # Test prediction functionality
    print(model.predict_next_action(input))
