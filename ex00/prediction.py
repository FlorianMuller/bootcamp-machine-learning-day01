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


if __name__ == "__main__":
    x = np.arange(1, 6)

    # Example 1:
    theta1 = np.array([5, 0])
    print(predict_(x, theta1))
    # Output:
    # array([5., 5., 5., 5., 5.])

    # Example 2:
    theta2 = np.array([0, 1])
    print(predict_(x, theta2))
    # Output:
    # array([1., 2., 3., 4., 5.])

    # Example 3:
    theta3 = np.array([5, 3])
    print(predict_(x, theta3))
    # Output:
    # array([ 8., 11., 14., 17., 20.])

    # Example 4:
    theta4 = np.array([-3, 1])
    print(predict_(x, theta4))
    # Output:
    # array([-2., -1., 0., 1., 2.])

    # 1D / 2D
    print(predict_(x, np.array([[-3], [1]])))
    print(predict_(x[:, None], theta4))
    print(predict_(x[:, None], np.array([[-3], [1]])))
    # Output:
    # (same as example 4)
    # array([-2., -1., 0., 1., 2.])

    # More than one x
    x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.],
                   [0.6, 6., 60.], [0.8, 8., 80.]])
    thetaX2 = np.array([[0.05], [1.], [1.], [1.]])
    print(predict_(x2, thetaX2))
    # Output:
    # array([22.25 44.45 66.65 88.85])

    # Empty array
    print(predict_(x, np.array([])))
    print(predict_(np.array([]), theta4))
    # Output:
    # None

    # Bad dimension
    print(predict_(x, np.array([[1, 1], [2, 2]])))
    print(predict_(np.array([[1, 1], [2, 2]]), theta4))
    # Output:
    # None
