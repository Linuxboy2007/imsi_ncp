import numpy as np
from scipy.sparse import diags

def phillipsmod(m, n, errorlevel):
    """
    Computes the data for the modified Phillips problem.

    Parameters:
    m (int): number of observations y (number of rows in the kernel matrix)
    n (int): number of unknown values of x to be determined (number of columns in the kernel matrix)
    errorlevel (float): defines the size of the random perturbations added to the right-hand side of the problem

    Returns:
    xstar (ndarray): true solution (n x 1)
    ystar (ndarray): "true" right-hand side, computed as Kstar * xstar
    K (ndarray): discretization of the kernel using trapezoidal rule integration (m x n)
    y (ndarray): ystar + random perturbations with mean 0 and standard deviation S
    Sdiag (ndarray): vector of diagonal elements of S (m x 1), with entries errorlevel * sqrt(ystar)
    A (ndarray): S^{-1} K
    b (ndarray): S^{-1} y
    t (ndarray): points at which the data y was measured (m x 1)
    xi (ndarray): points at which the solution y is desired (n x 1)
    """

    # Set some constants
    alpha0 = 0.1
    alpha = np.array([0.5, 0.5, 1])
    phi = np.array([-1.5, 0.5, 1.5])
    alimit = 5.9625

    # Set the points t and xi
    t = np.linspace(-alimit, alimit, m)
    xi = np.linspace(-3, 3, n)

    # Set the quadrature weights for trapezoidal rule
    hquad = 6 / (n - 1)
    w = hquad * np.ones(n)
    w[0] = w[0] / 2
    w[-1] = w[-1] / 2

    # Set the true solution xstar
    beta = alpha0 * (1 + np.cos(np.pi * xi / 3))
    beta = beta * (np.abs(xi) <= 3)

    c = np.zeros((n, 3))
    for k in range(3):
        c[:, k] = (1 + np.cos(2 * np.pi * (xi - phi[k])))
        c[:, k] = c[:, k] * (np.abs(xi - phi[k]) <= 0.5)

    xstar = beta + c @ alpha

    # Compute the kernel matrix K
    K = np.zeros((m, n))
    for i in range(m):
        Kval = (1 + np.cos((xi - t[i]) * np.pi / 3)) / 6
        Kval = Kval * (np.abs(xi - t[i]) <= 3)
        K[i, :] = Kval * w

    # Compute the right-hand side ystar
    ystar = K @ xstar

    # Compute Sdiag
    Sdiag = errorlevel * np.sqrt(ystar)
    zero_indices = ystar == 0
    if np.any(zero_indices):
        Sdiag[zero_indices] = np.finfo(float).eps

    # Add errors to ystar to obtain y
    y = ystar + Sdiag * np.random.randn(m)

    # Compute the matrix and right-hand side for the scaled problem
    Sinv = diags(1 / Sdiag, 0, shape=(m, m))
    A = Sinv @ K
    b = Sinv @ y

    return xstar, ystar, K, y, Sdiag, A, b, t, xi

xstar, ystar, K, y, Sdiag, A, b, t, xi = phillipsmod(10,5,1e-05)
