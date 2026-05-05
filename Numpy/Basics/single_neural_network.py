import numpy as np

def train_neuron(features: np.ndarray,
                 labels: np.ndarray,
                 initial_weights: np.ndarray,
                 initial_bias: float,
                 learning_rate: float, epochs: int) \
        -> (np.ndarray, float, list[float]):
    updated_weights = initial_weights[:, np.newaxis]
    updated_bias = initial_bias
    mse_values = np.zeros(epochs)
    samples = len(labels)
    labels = labels[:, np.newaxis]

    for i in range(epochs):
        # Forward pass
        z = features @ updated_weights + updated_bias
        a = 1 / (1 + np.exp(-z))

        # Compute loss
        mse_values[i] = np.mean(np.square(labels - a))

        # Backpropagation
        dl_da = 2 / samples * (a - labels)
        dl_dz = dl_da * a * (1 - a)

        dl_dw = features.T @ dl_dz
        dl_db = np.sum(dl_dz)

        # Weights update
        updated_weights -= learning_rate * dl_dw
        updated_bias -= learning_rate * dl_db


    return np.round(updated_weights.squeeze(), 4), np.round(updated_bias, 4), np.round(mse_values, 4)