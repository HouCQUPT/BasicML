import numpy as np
import matplotlib.pyplot as plt


class GD:
    """
    Gradient Descend
    """
    def __init__(self, x, y):
        """
        :type x: np.array
        :type y: np.array
        """
        self.x = x
        self.y = y
        self.m, self.n = np.shape(x)
        b = np.ones(shape=[self.m, 1])
        self.x = np.concatenate((x, b), axis=1)        # add bias
        self.w = np.random.rand(self.n+1, 1)                # weight vector

    def batch(self, lear_rate=0.00000001, max_iter=50000):
        x = self.x
        y = self.y
        w = self.w
        error_matrix = np.zeros(shape=[max_iter, 1])
        for iter in range(max_iter):
            error = np.dot(x, w) - y
            error_matrix[iter, 0] = np.log10(np.sum(error ** 2) / np.shape(x)[0])
            w -= lear_rate * (np.dot(x.T, error))
        plt.figure(0)
        plt.plot(range(max_iter), error_matrix, '-')
        plt.show()
        return w

    def mini_batch(self, lear_rate=0.0001, max_iter=5000):
        x = self.x
        y = self.y
        w = self.w
        error_matrix = np.zeros(shape=[max_iter, 1])
        _BATCH = 100
        for iter in range(max_iter):
            for i in range(0, _BATCH, np.shape(x)[0]):
                error = np.dot(x[i:i+_BATCH], w) - y[i:i+_BATCH]
                error_matrix[iter, 0] = np.log10(np.sum(error ** 2) / _BATCH)
                w -= lear_rate * (np.dot(x[i:i+_BATCH].T, error))
        plt.figure(0)
        plt.plot(range(max_iter), error_matrix, '-')
        plt.show()
        return w


if __name__ == "__main__":
    