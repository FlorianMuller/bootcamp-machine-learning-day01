import numpy as np


def predict_(x, theta):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if theta.ndim == 2 and theta.shape[1] == 1:
        theta = theta.flatten()

    if (x.size == 0 or theta.size == 0
        or x.ndim != 2 or theta.ndim != 1
            or theta.shape[0] != x.shape[1] + 1):
        return None

    # np.dot(a,b) if a is an N-D array and b is a 1-D array
    # => it is a sum product over the last axis of a and b.
    return np.dot(np.c_[np.ones(x.shape[0]), x], theta)


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    if theta.ndim == 2 and theta.shape[1] == 1:
        theta = theta.flatten()

    if (x.size == 0 or y.size == 0 or theta.size == 0
        or x.ndim != 2 or y.ndim != 1 or theta.ndim != 1
            or x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]):
        return None

    x_padded = np.c_[np.ones(x.shape[0]), x]
    new_theta = theta.astype("float64")
    for _ in range(max_iter):
        nabla = x_padded.T.dot(x_padded.dot(new_theta) - y) / y.shape[0]
        new_theta[0] = new_theta[0] - alpha * nabla[0]
        new_theta[1] = new_theta[1] - alpha * nabla[1]

    return new_theta


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1])

    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1)
    # Output:
    # array([[1.40709365],
    # [1.1150909 ]])

    # Example 1:
    print(predict_(x, theta1))
    # Output:
    # array([[15.3408728 ],
    # [25.38243697],
    # [36.59126492],
    # [55.95130097],
    # [65.53471499]])
