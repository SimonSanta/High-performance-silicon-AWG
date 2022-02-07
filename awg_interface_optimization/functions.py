# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:44:47 2020

@author: Simon Santacatterina
"""
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import BSpline

def turn_spline(X3,s):
    
    xnew = np.linspace(X3[:,0].min(), X3[:,0].max(), 100) 
    sp = make_interp_spline(X3[:,0], X3[:,s], k=2)  # type: BSpline
    power = sp(xnew)
    return xnew, power
