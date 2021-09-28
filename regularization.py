#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt

def rbf(X, cc, hh):
    X = np.array(X)
    return np.exp(-(X - cc)**2 / hh**2)

def phi_rbf(X, K, x_max):
    X_tild = np.ones(X.shape)
    for i in np.linspace(0, x_max, K-1):
        X_tild = np.hstack((X_tild, rbf(X, i, 120)))
    return X_tild

def reg_Phi(Phi, K, reg_c):
    rPhi = np.vstack((Phi, np.eye(K,K)*np.sqrt(reg_c)))
    return rPhi 

def reg_yy(yy, K):
    ryy = np.vstack((yy, np.zeros(K)[:,None]))
    return ryy


def main():
    #np.random.seed(seed=12)
    # Clear plot from previous runs
    plt.clf()

    # Dimensions
    N, D = 5, 1
    K = int(sys.argv[1]) if len(sys.argv) == 2 else 9
    x_max = 100

    # Standard non-regularization
    #X = x_max * np.random.rand(N,D)
    X = np.linspace(1, 100, N).reshape(N,D)
    #yy = 0.1* np.random.randn(N,D) + 3
    yy = np.log(X) + 0.1 * np.random.randn(N,D)
    
    Phi = phi_rbf(X, K, x_max)
    print(Phi)
    ww = np.linalg.lstsq(Phi, yy, rcond=0)[0]
    print(ww)
    
    X_grid = np.arange(0, x_max, 0.01)[:, None]
    oo = phi_rbf(X_grid, K, x_max) @ ww

    # Regularization
    reg_c = 0.01
    rPhi = reg_Phi(Phi, K, reg_c)
    ryy = reg_yy(yy, K)
    rww = np.linalg.lstsq(rPhi, ryy, rcond=None)[0]
    print(rww)
    roo = phi_rbf(X_grid, K, x_max) @ rww

    # Show plot
    plt.plot(X_grid, oo, 'b-')
    plt.plot(X_grid, roo, 'g-')
    plt.plot(X, yy, 'r*')
    plt.ylim(1,5)
    plt.show()



if __name__ == "__main__":
    main()
