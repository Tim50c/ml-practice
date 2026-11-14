# ml-practice

Machine learning algorithms and the math behind them, implemented **from scratch with NumPy**. The goal is to understand what happens inside the library calls (`sklearn`, `torch`) by rebuilding them from first principles: no ML frameworks, only NumPy (plus SciPy/Matplotlib for a few statistical helpers and plots).

## What's inside

### `Machine Learning Algorithms/`
| Area | Implementations |
|---|---|
| Tree-based | Decision tree classifier (Gini/entropy, sample/class weights, cost-complexity pruning, best-first growth), decision tree regressor, binary split classifiers |
| Ensembles | Bagging, majority voting, Random Forest, AdaBoost (decision stumps) |
| Linear models | Linear regression with L2 weight decay (Adam optimizer), softmax regression, Linear Discriminant Analysis |
| Instance / probabilistic | k-Nearest Neighbors, Gaussian and Bernoulli Naive Bayes |
| Preprocessing | PCA, SMOTE, missing-value imputation, stratified train/test split, sorted polynomial features |
| Metrics | ROC curve, precision-recall curve |

### `Autograd/`
A tiny scalar-valued reverse-mode automatic differentiation engine (`Value`, with backprop over a topologically sorted graph) and a small MLP (`Neuron`, `Layer`, `MLP`) built on top of it, in the spirit of micrograd.

### `Numpy/`
- **Linear Algebra**: Gaussian elimination, LU (Doolittle), QR, Gram-Schmidt, Jacobi method, SVD, positive-definiteness check, correlation matrix.
- **Probability and Statistics**: probability distributions, hypothesis testing, covariance from a joint PMF; information theory (entropy, cross-entropy, KL divergence for uni/multivariate normals, Jensen-Shannon, mutual information, ELBO); probabilistic models (Gaussian Mixture Model fitted with EM, Markov chain simulation).
- **Basics**: NumPy notes and exercises (broadcasting, indexing/masking, reshaping, aggregations, a single-neuron network, etc.).

## Setup

Requires Python 3.13+. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
pip install numpy scipy matplotlib
```

Most modules are standalone functions or classes meant to be imported and called with NumPy arrays. For example:

```python
import numpy as np
from decision_tree import DecisionTree  # Machine Learning Algorithms/Decision Tree/

X = np.random.randn(200, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

clf = DecisionTree(criterion="gini", max_depth=5, random_state=0)
clf.fit(X, y)
print(clf.predict(X[:5]))
```

## Notes

This is a learning repo. The code favors clarity and a close match to the underlying math over performance, and it has not been benchmarked against production libraries.
