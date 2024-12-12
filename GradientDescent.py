import numpy as np


def ComputeGradient(f, vector, h=1e-5):
    """
    Computes the gradient of a scalar function f at point x using finite differences.

    Parameters:
    - f: The function to differentiate.
    - vector: Point (scalar or array) at which to evaluate the gradient.
    - h: Step size for finite difference approximation.

    Returns:
    - grad: The gradient of f at x (same shape as x).
    """
    grad = np.zeros_like(vector)
    for i in range(len(vector)):
        vector_forward = np.copy(vector)
        vector_backward = np.copy(vector)
        vector_forward[i] += h
        vector_backward[i] -= h
        grad[i] = (f(*vector_forward) - f(*vector_backward)) / (2 * h)
    return grad


def GradientDescent(f, UB, LB, dimension, learning_rate=0.1, tolerance=1e-6, max_iterations=1000):
    """
    Performs gradient descent to minimize any convex function f.

    Parameters:
    - f: The function to minimize (must be differentiable).
    - InitVector: Initial values of x, y, z (can be a vector).
    - learning_rate: Step size for updates.
    - tolerance: Convergence threshold.
    - max_iterations: Maximum number of iterations.

    Returns:
    - x_min: The value of x that minimizes f.
    - f_min: The minimum value of f(x).
    - iterations: Number of iterations taken to converge.
    """
    InitVector = np.random.uniform(LB, UB, size=dimension)
    vector = np.array(InitVector, dtype=np.float64)
    iterations = 0

    while iterations < max_iterations:
        grad = ComputeGradient(f, vector)
        NewVector = vector - learning_rate * grad
        if np.linalg.norm(NewVector - vector) < tolerance:
            break

        vector = NewVector
        iterations += 1

    f_min = f(*vector)
    return vector, f_min, iterations

