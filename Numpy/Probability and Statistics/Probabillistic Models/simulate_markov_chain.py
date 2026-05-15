import numpy as np


def simulate_markov_chain(
    transition_matrix: np.ndarray,
    initial_state: int,
    num_steps: int,
) -> np.ndarray:
    """
    Simulate a discrete-time Markov chain.

    Args:
        transition_matrix:
            Square transition probability matrix where
            transition_matrix[i, j] is P(state_j | state_i).
        initial_state:
            Starting state index.
        num_steps:
            Number of transitions to simulate.
        rng:
            Optional NumPy random number generator.

    Returns:
        Array of visited states of length num_steps + 1.
    """
    P = transition_matrix

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(
            "transition_matrix must be a square matrix"
        )

    n_states = P.shape[0]

    if not (0 <= initial_state < n_states):
        raise ValueError("initial_state is out of bounds")

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")

    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError(
            "Each row of transition_matrix must sum to 1"
        )

    if np.any(P < 0):
        raise ValueError(
            "transition probabilities must be non-negative"
        )

    rng = np.random.default_rng()

    states = np.empty(num_steps + 1, dtype=np.int64)
    states[0] = initial_state

    current_state = initial_state

    state_indices = np.arange(n_states)

    for step in range(1, num_steps + 1):
        current_state = rng.choice(
            state_indices,
            p=P[current_state],
        )

        states[step] = current_state

    return states

transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
print(simulate_markov_chain(transition_matrix, 0, 2))
