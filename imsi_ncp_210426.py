#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:03:37 2026

@author: linuxboy
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# import src.utilities as util
import src.iterative as iterative
import src.problems as prb


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', SyntaxWarning)

m = 2048
n = 1024
# A, f, u_ext = prb.phillips(m)

u_ext, f, K, bn, Sdiag, A, fn, _, _ = prb.phillipsmod(m, n, 1e-05)
x = np.linspace(-3,3, n)
u_ext = u_ext.reshape((-1,1))
f = f.reshape((-1,1))
fn = fn.reshape((-1,1))
delta = LA.norm(f-fn)
# u_ext, f, A, _, _, _, _, _, _ = prb.phillipsmod(m,n,1e-06)
# x = np.linspace(-3,3, n)
# u_ext = u_ext.reshape((-1,1))
# f = f.reshape((-1,1))
# noise_value = 1e-05
# fn, delta = util.add_noise(f, noise_value, 1)
print('Matrix size', m, n)

k = 500
k_ncp = 500

tau = 1 + np.sqrt(np.finfo(np.float64).eps)
smax_full = np.sqrt(LA.norm(A,np.inf)*LA.norm(A,1))
omega = smax_full/k
omega_ncp = smax_full/k_ncp

'''
all calculations are performed in double precision
'''
xk, it, itp, tk, tp = iterative.imsi_bis(A, fn, omega, delta, 1e-01, tau)
print('Неявный метод простых итераций на основе псевдообращения')
print('Двойная точность')
print('Число итераций: %d' % it)
print('Число итераций BI: %d' % itp)
print('Время(sec): %.2f' % tk)
Rerr0 = LA.norm(u_ext - xk) / LA.norm(u_ext)
print('Относительная ошибка %.2e' % Rerr0)
plt.plot(x, u_ext,'r', x, xk,'b--')
plt.legend(('true', 'x_k'), loc='best')
plt.show()

xk_ncp, it_ncp, itp_ncp, tk_ncp, tp_ncp = iterative.imsi_bisncp(A, fn, omega_ncp, 1e-01)
print('Неявный метод простых итераций на основе псевдообращения (ncp)')
print('Двойная точность')
print('Число итераций: %d' % it_ncp)
print('Число итераций BI: %d' % itp_ncp)
print('Время(sec): %.2f' % tk_ncp)
Rerr_ncp = LA.norm(u_ext - xk_ncp) / LA.norm(u_ext)
print('Относительная ошибка %.2e' % Rerr_ncp)

plt.plot(x, u_ext,'r', linewidth = 0.5, label='true')
plt.plot(x, xk_ncp,'b--', linewidth = 0.75, label='x_k(NCP)')
plt.legend(('true', 'x_k(NCP)'), loc='best')
plt.show()
