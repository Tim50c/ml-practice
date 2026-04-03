import numpy as np

# Activation functions
def sigmoid(x):
    """"Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """"ReLU activation function"""
    return np.maximum(0, x)

def softmax(x):
    """"Softmax for multi-class classification"""
    exp_x = np.exp(x - np.max(x))   # Numerical Stability
    return exp_x / (np.sum(exp_x, axis=-1, keepdims=True))

# One-hot encoding
def one_hot_encoding(y, n_class):
    """"One-hot encoding for labels"""
    one_hot = np.zeros((len(y), n_class))
    one_hot[np.arange(len(y)), y] = 1

    return one_hot

# Neural network forward pass
def forward_pass(X, W1, b1, W2, b2):
    """
    X: Input (n_samples, n_features)
    W1: First layer weights (n_features, n_hidden)
    b1: First layer bias (n_hidden,)
    W2: Second layer weights (n_hidden, n_output)
    b2: Second layer bias (n_output,)
    """
    # Layer 1
    z1 = X @ W1 + b1
    a1 = relu(z1)

    # Layer 2 (output)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)

    return a2