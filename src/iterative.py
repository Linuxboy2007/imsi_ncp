# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:41:29 2026

@author: linuxboy
"""

import numpy as np
from numpy import linalg as LA
import time as time
import src.utilities as util

def ipinv(A, m, epsp):
    r"""
    Parameters
    ----------
    A : (m, n) np.ndarray
        The input matrix.
    epsp : double
        epsilon for the stopping criterion.

    Returns
    -------
    X: (n, m) np.ndarray
        the pseudo inverse of input matrix A.
    it : int
        number of interations.

    """
    n = A.shape[1]
    beta = 1.8/LA.norm(A,'fro')**2
    Em = 2*np.eye(n)
    X = beta*A
    Xold = X
    A = A.T
    err = 1e+09;it = 0
    while err > epsp:
           X = Xold@(Em - A@Xold)
           err = LA.norm(X - Xold,np.inf)/LA.norm(Xold,np.inf)
           Xold = X
           it = it + 1
           
    return X[0:m,0:n].T, it

def imsi_bis(A, f, omega, delta, epsp, tau = 1.01):
    start = time.time() 
    m, n = A.shape
    xk = np.zeros([n, 1])
    Aw = np.row_stack((A, omega*np.eye(n)))
    startp = time.time() 
    Ua, itp = ipinv(Aw, m, epsp)
    endp = time.time() - startp
    it = 0; err = 1e+10
    while err >= (tau*delta):
        r = f - A@xk
        xk = xk + Ua@r
        err = LA.norm(r)
        it = it +1
    xk = xk.reshape(-1,1)
    end = time.time() - start
    return xk, it, itp, end, endp

def imsi_bisncp(A, f, omega, epsp):
    start = time.time() 
    m, n = A.shape
    xk = np.zeros([n, 1])
    Aw = np.row_stack((A, omega*np.eye(n)))
    startp = time.time() 
    Ua, itp = ipinv(Aw, m, epsp)
    endp = time.time() - startp
    it = 0; err = 1e+10
    q = np.floor(m / 2).astype(int) if np.isrealobj(f) else m - 1
    kq = 1.36*q**(-0.5);
    # kq = 1.63*q**(-0.5);
    print('kq: %.3e' % kq)
    while err >= kq:
        r = f - A@xk
        xk = xk + Ua@r
        cp, err = util.ncp(r)
        print('err: %.3e' % err)
        it = it +1
    xk = xk.reshape(-1,1)
    end = time.time() - start
    return xk, it, itp, end, endp

def imsi_svd(u, s, vh, f, omega, delta, tau = 1.01):
    start = time.time() 
    n = s.shape[0]
    n1 = vh.shape[1]
    s2w = s.reshape(n,1)**2+omega**2
    ga = vh[:,0:n1].T@np.divide(np.multiply(s.reshape(n,1),u[:,0:n].T@f),s2w)
    ga1 = vh[:,0:n1].T@np.divide(vh[:,0:n1],s2w)
    theta = np.zeros((vh.shape[1],1))
    ee = s*np.eye(n)
    it = 0; err = 1e+10
    while err >= tau*delta:
        theta = omega**2*ga1@theta + ga
        it = it +1
        
        err = LA.norm((u@(ee@vh))@theta - f)
    end = time.time() - start
    return theta.reshape(n1,1), it, end