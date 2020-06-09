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


def simple_gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible dimensions.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a 2 * 1 vector.
    Returns:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    y_hat = predict_(x, theta)

    if (x.size == 0 or y.size == 0 or theta.size == 0
        or x.shape != y.shape or theta.shape != (2,)
            or y_hat is None):
        return None

    # print("y_hat - y", y_hat - y)
    # print("(y_hat - y) * x", (y_hat - y) * x)
    nabla0 = np.sum(y_hat - y) / y.shape[0]
    nabla1 = np.sum((y_hat - y) * x) / y.shape[0]

    return np.array([nabla0, nabla1])
    # return np.abs(np.array([nabla0, nabla1]))


if __name__ == "__main__":
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])

    # Example 0:
    theta1 = np.array([2, 0.7])
    print(simple_gradient(x, y, theta1))
    # Output:
    # subject: array([21.0342574,  587.36875564])
    # me:          [ -19.0342574  -586.66875564]

    # Example 1:
    theta2 = np.array([1, -0.4])
    print(repr(simple_gradient(x, y, theta2)))
    # Output:
    # subject: array([58.86823748,  2229.72297889])
    # me:          [ -57.86823748  -2230.12297889]
