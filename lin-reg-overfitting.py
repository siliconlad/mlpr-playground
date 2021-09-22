#!/usr/bin/env python3

import numpy as np

def noise(X, sigma):
    return X + np.random.normal(0, sigma, X.shape)

N, D = 100, 1
sigma = 0.5

rounds = 10000
total_error = 0
total_diff = 0
for i in range(rounds):
    # Generate X (features)
    X = np.random.randn(N,D)

    # Generate noisy observations
    yy = noise(X**2, sigma)

    # polynomial basis transform
    phi = np.hstack((X**2,))
    #phi = np.hstack((X**3, X**2, X, np.ones((N,D))))
    #phi = np.hstack((X**4, X**3, X**2, X, np.ones((N,D))))
    #phi = np.hstack((X**5, X**4, X**3, X**2, X, np.ones((N,D))))
    #phi = np.hstack((X**6, X**5, X**4, X**3, X**2, X, np.ones((N,D))))

    # fit with least squared
    ww, error, _, _ = np.linalg.lstsq(phi, yy, rcond=None)

    # Keep track of errors
    total_error += error
    total_diff += np.sum((X**2 - yy)**2)


exp_error = N * sigma**2
print("Exp error: ", exp_error)

avg_error = total_error[0] / rounds
print("Avg error: ", avg_error)

print("Avg diff:  ", total_diff / rounds)

stds = (avg_error - exp_error) / (sigma**2 * np.sqrt(2 * N))
print("No. std:   ", stds)
