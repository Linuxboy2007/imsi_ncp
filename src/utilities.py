#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:40:15 2026

@author: linuxboy
"""

import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt


def add_noise(f, noise_level,tn = 1):
    if tn == 1:
        noise = np.random.randn(f.shape[0],1)
        noise = (LA.norm(f)*noise_level*noise)/LA.norm(noise)
        fn = f + noise
        delta = LA.norm(f - fn)
    else:
        noise = 2* np.random.rand(f.shape[0], 1) - 1
        fn = f + noise_level*LA.norm(f)*(noise/LA.norm(noise))
        fn = fn.reshape((-1,1))
        delta = LA.norm(f - fn)
    return fn, delta

'''
ncp
'''
def ncp(r):
    m = len(r)   
    # Determine q based on whether beta is real
    q = np.floor(m / 2).astype(int) if np.isrealobj(r) else m - 1
    
    # Compute periodogram (squared magnitude of FFT)
    D = np.abs(np.fft.fft(r))**2
    D = D[1:q+1]  # Python uses 0-based indexing (equivalent to MATLAB's 2:q+1)
    
    v = np.arange(1, q + 1) / q  # Uniform distribution reference
    cp = np.cumsum(D) / np.sum(D)  # Normalized cumulative periodogram
    
    dist = np.linalg.norm(cp - v, np.inf)  # Distance metric

    return cp, dist

def ncp_plot(r, lv = 95):
    cp, dist = ncp(r)
    q = cp.shape[0]
    
    if lv == 95:
        kq = 1.36*q**(-0.5);
    else:
        kq = 1.63*q**(-0.5);
    
    xq = np.arange(1, q+1, 1)
    wn = xq/q
    xp = wn + kq
    xm = wn - kq
    plt.plot(xq, xm,'k--',xq,xp,'k--', xq, wn,'r-', xq, cp,'b-')
    plt.title('normalized cumulative periodogram')
    plt.show()
    
    # return cp
    # , wn, xq, xp, xm
