#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:43:32 2026

@author: linuxboy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 22:47:37 2025

@author: linuxboy
"""

import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse import diags


def shaw(n):
    """
    SHAW Test problem: one-dimensional image restoration model.
    
    Parameters
    ----------
    n : int
        The order of the problem, must be even.
    
    Returns
    -------
    A : ndarray
        The n x n matrix representing the discretized kernel.
    b : ndarray
        The right-hand side vector (A @ x).
    x : ndarray
        The solution vector.
    
    Notes
    -----
    Discretization of a first kind Fredholm integral equation with
    [-pi/2, pi/2] as both integration intervals.
    """
    
    # Check input
    if n % 2 != 0:
        raise ValueError('The order n must be even')
    
    # Initialization
    h = np.pi / n
    A = np.zeros((n, n))
    
    # Compute the matrix A
    co = np.cos(-np.pi/2 + np.arange(0.5,n) * h)
    psi = np.pi * np.sin(-np.pi/2 + np.arange(0.5,n) * h)
    
    for i in range(n//2):
        for j in range(i, n - i):
            ss = psi[i] + psi[j]
            if ss == 0:
                # Handle the singularity when ss=0 (using limit as u->0)
                A[i, j] = (co[i] + co[j])**2
            else:
                A[i, j] = ((co[i] + co[j]) * np.sin(ss) / ss)**2
            # A[i, j] = ((co[i] + co[j]) * np.sin(ss) / ss)**2
            A[n - j - 1, n - i - 1] = A[i, j]
        A[i, n - i - 1] = (2 * co[i])**2
    
    A = A + np.triu(A, 1).T
    A = A * h
    
    # Compute the vectors x and b
    a1 = 2; c1 = 6; t1 =  0.8
    a2 = 1; c2 = 2; t2 = -0.5
    
    t = -np.pi/2 + np.arange(0.5,n) * h
    x = (a1 * np.exp(-c1 * (t - t1)**2) + a2 * np.exp(-c2 * (t - t2)**2))
    b = A @ x
    
    return A, b, x

def shawmod(n, errorlevel = 1e-05):
    """
    SHAW Test problem: one-dimensional image restoration model.
    
    Parameters
    ----------
    n : int
        The order of the problem, must be even.
    
    Returns
    -------
    A : ndarray
        The n x n matrix representing the discretized kernel.
    b : ndarray
        The right-hand side vector (A @ x).
    x : ndarray
        The solution vector.
    
    Notes
    -----
    Discretization of a first kind Fredholm integral equation with
    [-pi/2, pi/2] as both integration intervals.
    """
    
    # Check input
    if n % 2 != 0:
        raise ValueError('The order n must be even')
    
    # Initialization
    h = np.pi / n
    A = np.zeros((n, n))
    
    # Compute the matrix A
    co = np.cos(-np.pi/2 + np.arange(0.5,n) * h)
    psi = np.pi * np.sin(-np.pi/2 + np.arange(0.5,n) * h)
    
    for i in range(n//2):
        for j in range(i, n - i):
            ss = psi[i] + psi[j]
            if ss == 0:
                # Handle the singularity when ss=0 (using limit as u->0)
                A[i, j] = (co[i] + co[j])**2
            else:
                A[i, j] = ((co[i] + co[j]) * np.sin(ss) / ss)**2
            # A[i, j] = ((co[i] + co[j]) * np.sin(ss) / ss)**2
            A[n - j - 1, n - i - 1] = A[i, j]
        A[i, n - i - 1] = (2 * co[i])**2
    
    A = A + np.triu(A, 1).T
    A = A * h
    
    # Compute the vectors x and b
    a1 = 2; c1 = 6; t1 =  0.8
    a2 = 1; c2 = 2; t2 = -0.5
    
    t = -np.pi/2 + np.arange(0.5,n) * h
    x = (a1 * np.exp(-c1 * (t - t1)**2) + a2 * np.exp(-c2 * (t - t2)**2))
    f = A @ x
    
    Sdiag = errorlevel * np.sqrt(f)
    zero_indices = f == 0
    if np.any(zero_indices):
        Sdiag[zero_indices] = np.finfo(float).eps

    # Add errors to ystar to obtain y
    y = f + Sdiag * np.random.randn(n)

    # Compute the matrix and right-hand side for the scaled problem
    Sinv = diags(1 / Sdiag, 0, shape=(n, n))
    A = Sinv @ A
    b = Sinv @ y
    
    return A, b, x, y

def gravity(n, example=None, a=None, b=None, d=None):
    """
    GRAVITY Test problem: 1-D gravity surveying model problem
    
    [A, b, x] = gravity(n, example, a, b, d)
    
    Discretization of a 1-D model problem in gravity surveying, in which
    a mass distribution f(t) is located at depth d, while the vertical
    component of the gravity field g(s) is measured at the surface.
    
    Parameters:
    -----------
    n : int
        Number of discretization points
    example : int, optional (default=1)
        Example number (1, 2, or 3)
    a : float, optional (default=0)
        Start of measurement interval
    b : float, optional (default=1)
        End of measurement interval
    d : float, optional (default=0.25)
        Depth of the magnetic deposit
        
    Returns:
    --------
    A : ndarray
        The n x n matrix
    b : ndarray
        The right-hand side vector (A @ x)
    x : ndarray
        The solution vector
    """
    
    # Set default values
    if example is None:
        example = 1
    if a is None:
        a = 0
    if b is None:
        b = 1
    if d is None:
        d = 0.25
    
    # Set up abscissas and matrix
    dt = 1 / n
    ds = (b - a) / n
    t = dt * (np.arange(1, n+1) - 0.5)
    s = a + ds * (np.arange(1, n+1) - 0.5)
    
    T, S = np.meshgrid(t, s)
    A = dt * d * np.ones((n, n)) / (d**2 + (S - T)**2)**(3/2)
    
    # Set up solution vector
    nt = round(n / 3)
    nn = round(n * 7 / 8)
    x = np.ones(n)
    
    if example == 1:
        x = np.sin(np.pi * t) + 0.5 * np.sin(2 * np.pi * t)
    elif example == 2:
        x[:nt] = (2/nt) * np.arange(1, nt+1)
        x[nt:nn] = ((2*nn - nt) - np.arange(nt+1, nn+1)) / (nn - nt)
        x[nn:] = (n - np.arange(nn+1, n+1)) / (n - nn)
    elif example == 3:
        x[:nt] = 2 * np.ones(nt)
    else:
        raise ValueError('Illegal value of example')
    
    b = A @ x
    
    return A, b, x

def heat(n, kappa=1):
    """
    Test problem: inverse heat equation (Python version of MATLAB's heat function)
    
    Parameters:
    n : int
        Size of the problem
    kappa : float, optional
        Controls conditioning of the matrix (default=1)
        kappa=5 gives well-conditioned problem
        kappa=1 gives ill-conditioned problem
    
    Returns:
    A : ndarray
        The generated matrix
    b : ndarray
        Right-hand side vector (if returned)
    x : ndarray
        Exact solution vector (if returned)
    """
    
    # Initialization
    h = 1 / n
    t = np.arange(h/2, 1 + h/2, h)  # Equivalent to h/2:h:1 in MATLAB
    
    c = h / (2 * kappa * np.sqrt(np.pi))
    d = 1 / (4 * kappa**2)
    
    # Compute the matrix A
    k = c * t**(-1.5) * np.exp(-d / t)
    r = np.zeros_like(t)
    r[0] = k[0]
    A = toeplitz(k, r)
    
    # Compute vectors x and b if requested
    x = np.zeros(n)
    for i in range(n//2):
        ti = (i+1) * 20 / n  # MATLAB uses 1-based indexing
        if ti < 2:
            x[i] = 0.75 * ti**2 / 4
        elif ti < 3:
            x[i] = 0.75 + (ti - 2) * (3 - ti)
        else:
            x[i] = 0.75 * np.exp(-(ti - 3) * 2)
        
        # The second half is zeros
    x[n//2:] = 0
    b = A @ x  # Matrix-vector multiplication
        
    return A, b, x

def deriv2(n, example=1):
    """
    DERIV2 Test problem: computation of the second derivative.
    
    [A, b, x] = deriv2(n, example)
    
    This is a mildly ill-posed problem. It is a discretization of a
    first kind Fredholm integral equation whose kernel K is the
    Green's function for the second derivative.
    
    Parameters:
    -----------
    n : int
        The size of the problem.
    example : int, optional (default=1)
        Example number to choose between different right-hand sides
        and solutions.
        
    Returns:
    --------
    A : ndarray
        The n x n matrix.
    b : ndarray, optional
        The right-hand side vector (if nargout > 1).
    x : ndarray, optional
        The solution vector (if nargout == 3).
    """
    
    # Initialization.
    h = 1.0 / n
    sqh = np.sqrt(h)
    h32 = h * sqh
    h2 = h**2
    sqhi = 1.0 / sqh
    t = 2.0 / 3.0
    A = np.zeros((n, n))
    
    # Compute the matrix A.
    for i in range(n):
        A[i, i] = h2 * (( (i+1)**2 - (i+1) + 0.25 ) * h - ( (i+1) - t ))  # MATLAB uses 1-based indexing
        for j in range(i):
            A[i, j] = h2 * (j + 0.5) * ( (i + 0.5)*h - 1 )  # +0.5 because Python is 0-based
    
    A = A + np.tril(A, -1).T
    
    # Initialize outputs
    b = None
    x = None
    
    # Compute the right-hand side vector b if needed
    if True:  # In Python, we don't have nargout, so we'll always compute b and x
        b = np.zeros(n)
        if example == 1:
            for i in range(n):
                b[i] = h32 * (i + 0.5) * ( ((i+1)**2 + i**2) * h2 / 2 - 1 ) / 6
        elif example == 2:
            ee = 1 - np.exp(1)
            for i in range(n):
                b[i] = sqhi * (np.exp((i+1)*h) - np.exp(i*h) + ee * (i + 0.5) * h2 - h)
        elif example == 3:
            if n % 2 != 0:
                raise ValueError('Order n must be even')
            else:
                for i in range(n//2):
                    s12 = ((i+1)*h)**2
                    s22 = (i*h)**2
                    b[i] = sqhi * (s12 + s22 - 1.5) * (s12 - s22) / 24
                for i in range(n//2, n):
                    s1 = (i+1)*h
                    s12 = s1**2
                    s2 = i*h
                    s22 = s2**2
                    b[i] = sqhi * (-(s12 + s22)*(s12 - s22) + 4*(s1**3 - s2**3) - 
                            4.5*(s12 - s22) + h) / 24
        else:
            raise ValueError('Illegal value of example')
    
    # Compute the solution vector x if needed
    x = np.zeros(n)
    if example == 1:
        for i in range(n):
            x[i] = h32 * (i + 0.5)
    elif example == 2:
        for i in range(n):
            x[i] = sqhi * (np.exp((i+1)*h) - np.exp(i*h))
    elif example == 3:
        for i in range(n//2):
            x[i] = sqhi * (((i+1)*h)**2 - (i*h)**2) / 2
        for i in range(n//2, n):
            x[i] = sqhi * (h - (((i+1)*h)**2 - (i*h)**2) / 2)
    
    return A, b, x

def phillips(n):
    """
    PHILLIPS Test problem: Phillips' "famous" problem in Python.
    
    [A, b, x] = phillips(n)
    
    Discretization of the 'famous' first-kind Fredholm integral
    equation devised by D. L. Phillips.
    
    Parameters:
    -----------
    n : int
        The order of the problem, must be a multiple of 4.
    
    Returns:
    --------
    A : ndarray
        The n x n matrix.
    b : ndarray, optional
        The right-hand side vector of length n.
    x : ndarray, optional
        The solution vector of length n.
    
    Notes:
    ------
    The order n must be a multiple of 4.
    """
    
    # Check input
    if n % 4 != 0:
        raise ValueError('The order n must be a multiple of 4')
    
    # Compute the matrix A
    h = 12 / n
    n4 = n // 4
    r1 = np.zeros(n)
    
    c = np.cos(np.arange(-1, n4 + 1) * 4 * np.pi / n)
    r1[:n4] = h + 9 / (h * np.pi**2) * (2 * c[1:n4 + 1] - c[:n4] - c[2:n4 + 2])
    r1[n4] = h / 2 + 9 / (h * np.pi**2) * (np.cos(4 * np.pi / n) - 1)
    
    A = toeplitz(r1)
    # A = np.zeros((n, n))
    # for i in range(n):
    #     for j in range(n):
    #         A[i, j] = r1[abs(i - j)]
    
    # Compute the right-hand side b
    b = np.zeros(n)
    c_val = np.pi / 3
        
    for i in range(n // 2, n):
        t1 = -6 + (i + 1) * h  # MATLAB uses 1-based indexing
        t2 = t1 - h
        term1 = t1 * (6 - abs(t1) / 2)
        term2 = ((3 - abs(t1) / 2) * np.sin(c_val * t1) - 2 / c_val * (np.cos(c_val * t1) - 1)) / c_val
        term3 = t2 * (6 - abs(t2) / 2)
        term4 = ((3 - abs(t2) / 2) * np.sin(c_val * t2) - 2 / c_val * (np.cos(c_val * t2) - 1)) / c_val
        b[i] = term1 + term2 - term3 - term4
        b[n - i - 1] = b[i]  # Python uses 0-based indexing
        
    b = b / np.sqrt(h)
    
    # Compute the solution x
    x = np.zeros(n)
    c_val = np.pi / 3
    t = np.arange(0, 3 + 10 * np.finfo(float).eps, h)
    x[2 * n4 : 3 * n4 ] = (h + np.diff(np.sin(t * c_val)) / c_val) / np.sqrt(h)
    x[n4 : 2 * n4] = x[3 * n4 - 1 : 2 * n4 - 1 : -1]  # Reversed slice
    
    return A, b, x

def spectra(n, rho = 2,sigma = 4):
    A = np.zeros([n, n])
    x = np.zeros([n,1])
    tmp = 1/(rho*np.sqrt(2*np.pi))
    for i in range(n):
        for j in range(n):
            A[i,j] = tmp*np.exp((-1)*((i+1)-(j+1))**2/(2*rho**2))
    sc = round(n/68)
    # M0 = sc*22
    M1 = sc*30
    M2 = sc*38
    # M3 = sc*46
    a = 0.09974
    # 0.09974
    for i in range(n):
        # x[i] = a*np.exp(-(1)*((i+1)-M0)**2/sigma**2)+0.7*a*np.exp(-(1)*((i+1)-M1)**2/sigma**2) + 0.6*a*np.exp(-(1)*((i+1)-M2)**2/sigma**2)+0.55*a*np.exp(-(1)*((i+1)-M3)**2/sigma**2)
        x[i] = a*np.exp(-(1)*((i+1)-M1)**2/sigma**2) + a*np.exp(-(1)*((i+1)-M2)**2/sigma**2)
        
    # h = np.pi / n
    # # Compute the vectors x and b
    # a1 = 2; c1 = 6; t1 =  0.8
    # a2 = 1; c2 = 2; t2 = -0.5
    
    # t = -np.pi/2 + np.arange(0.5,n) * h
    # x = (a1 * np.exp(-c1 * (t - t1)**2) + a2 * np.exp(-c2 * (t - t2)**2))
    b = A @ x
    return A, b, x

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
    
    