import numpy as np


class NaiveBayes:
    w: np.ndarray | None
    b: np.ndarray | None
    single_class: int | None

    def __init__(self, smoothing=1.0):
        self.alpha = smoothing
        self.w = None
        self.b = None
        self.single_class = None

    def forward(self, X, y):
        classes = np.unique(y)
        if len(classes) == 1:
            self.single_class = classes[0]
            return

        self.single_class = None

        k = 2

        n, d = X.shape
        self.w = np.zeros((d, k), dtype=float)
        self.b = np.zeros((k,), dtype=float)

        mask = y[:, None] == np.arange(k)[None, :]
        N_c = np.sum(mask, axis=0)
        N_jc = X.T @ mask

        pi_c = N_c / n
        p_jc = (N_jc + self.alpha) / (N_c + 2*self.alpha)

        l = np.log(pi_c)
        L = np.log(p_jc)
        M = np.log(1 - p_jc)

        self.w = L - M
        self.b = l + np.ones((d,), dtype=float) @ M

    def predict(self, X):
        if self.single_class is not None:
            return np.full(X.shape[0], self.single_class)

        probs = X @ self.w + self.b
        return np.argmax(probs, axis=1)

model = NaiveBayes(smoothing=1.0)
X = np.array([[0, 0], [1, 0], [0, 1]])
y = np.array([0, 0, 0])
model.forward(X, y)
print(model.predict(np.array([[1, 1]])))
