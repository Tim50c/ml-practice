import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def generate_dataset(n_samples=100, n_classes=4, seed=42):
	"""Generate a synthetic dataset with 2 features and clear clusters."""
	np.random.seed(seed)
	centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]])
	X = np.random.randn(n_samples, 2) * 1.5
	y = np.random.randint(0, n_classes, n_samples)
	for i in range(n_samples):
		X[i] += centers[y[i]]
	return X, y


def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int):
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	Returns (losses, param_history) where param_history is a list of (W, b) tuples per iteration.
	"""
	n_samples, n_features = X.shape

	c = np.max(y) + 1
	y_one_hot = np.zeros((n_samples, c), dtype=int)
	y_one_hot[np.arange(n_samples), y] = 1

	W = np.full((n_features, c), 10, dtype=float)
	b = np.full((1, c), 10, dtype=float)

	losses = []
	param_history = []

	for _ in range(iterations):
		z = X @ W + b

		e_z = np.exp(z)
		p = e_z / np.sum(e_z, axis=1, keepdims=True)

		loss = -np.sum(y_one_hot * np.log(p + 1e-15))
		losses.append(loss)

		grad_W = X.T @ (p - y_one_hot)
		grad_b = np.sum(p - y_one_hot, axis=0)

		W = W - learning_rate*grad_W
		b = b - learning_rate*grad_b

		param_history.append((W.copy(), b.copy()))

	return losses, param_history


def visualize_softmax(X, y, losses, param_history, slow_iters=30, slow_pause=1.0, fast_pause=0.02):
	"""Animate decision regions and loss curve across training iterations."""
	n_classes = np.max(y) + 1
	markers = ['^', 's', 'o', 'D']
	colors = ['#FF4444', '#44CC44', '#4444FF', '#CCAA00']
	region_cmap = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF', '#FFFFAA'])

	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(
		np.linspace(x_min, x_max, 200),
		np.linspace(y_min, y_max, 200)
	)
	grid = np.c_[xx.ravel(), yy.ravel()]

	plt.ion()
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

	for i, (W, b) in enumerate(param_history):
		ax1.clear()
		ax2.clear()

		z = grid @ W + b
		pred = np.argmax(z, axis=1).reshape(xx.shape)
		ax1.contourf(xx, yy, pred, alpha=0.4, cmap=region_cmap,
		             levels=np.arange(n_classes + 1) - 0.5)

		for c in range(n_classes):
			mask = y == c
			ax1.scatter(X[mask, 0], X[mask, 1], marker=markers[c], s=40,
			            color=colors[c], edgecolors='black', linewidth=0.5,
			            label=f'Class {c}')
		ax1.set_xlim(x_min, x_max)
		ax1.set_ylim(y_min, y_max)
		ax1.set_title(f'Decision Regions (Iteration {i + 1}/{len(param_history)})')
		ax1.legend(loc='upper right', fontsize=8)

		ax2.plot(range(1, i + 2), losses[:i + 1], 'b-', linewidth=1.5)
		ax2.set_title('Cross Entropy Loss')
		ax2.set_xlabel('Iteration')
		ax2.set_ylabel('Loss')
		ax2.set_xlim(1, len(param_history))

		pause = slow_pause if i < slow_iters else fast_pause
		plt.pause(pause)

	plt.ioff()
	plt.show()


if __name__ == '__main__':
	X, y = generate_dataset(n_samples=150, n_classes=4)
	losses, param_history = train_softmaxreg(X, y, learning_rate=0.01, iterations=50)
	print(f'Initial loss: {losses[0]:.2f}, Final loss: {losses[-1]:.2f}')
	visualize_softmax(X, y, losses, param_history)
