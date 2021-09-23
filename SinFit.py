# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 19:12:24 2021

@author: Edwar
"""

import numpy as np
import matplotlib.pyplot as plt

def transform(xx):
    phi = np.vstack((np.ones(len(xx)),xx**1,xx**2,xx**3,xx**4,xx**5,xx**6,xx**7,xx**8,xx**9))
    return phi.T

datapoints = 1001
sigma = 0.1
omega = 3
x_max = 1
xx = np.linspace(0,x_max,datapoints)
y_true = np.sin(2*np.pi*omega*xx)
y_observed = y_true + np.random.normal(0,sigma, size = (datapoints))
phi = transform(xx)
ww = np.linalg.lstsq(phi, y_observed,rcond = None)[0]
predicts = phi@ww
plt.plot(xx,predicts)
plt.plot(xx,y_true)
