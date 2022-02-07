# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:33:21 2021

@author: Simon Santacatterina


Fraunhofer diffraction approximation for the computation of what happens
when the beams diffract in the slab. The parameters passed as arguments are:
    

"""
import numpy as np

"""

"""
def fraunhofer(dwg, n, r, wavelength, k, nx, ny, dx, dy, nwg_val, nh_val, x, y, xp, yp, phi_m):
    
    phi_p = np.zeros(shape=(nwg_val, nh_val),dtype=np.complex_) #each elment is a complex
    
    ix1 = np.arange(nwg_val)
    iy1 = np.arange(nh_val)
    
    #index_intx = np.arange(nx)
    index_inty  = np.arange(ny)
    
    for i in ix1:
        for j in iy1:
            
            rho = np.sqrt(r**2+ xp[i]**2 + yp[j]**2) #the distance 
            s1 = xp[i]/rho #coordinate of x in the arrival plane
            s2 = yp[j]/rho #coordinate of y in the arrival plane
            K = (1j/(wavelength*rho))*np.exp(-1j*k) #intergal constant
            
            count = 0
            
            #discrete intergral in 2D
            for p in index_inty:
                count = count + np.sum(phi_m[:,p]*np.exp((1j*k)*(s1*x[:]*10**-6+s2*y[p]*10**-6)))
            phi_p[i,j] = K*count*dx*dy #multiplied by the discretization windows in both directions
            
    return phi_p #the matrix is complex

"""
            for s in index_intx:
                for p in index_inty:
                    count = count + np.sum(phi_m[s,p]*np.exp((1j*k)*(s1*x[s]*10**-6+s2*y[p]*10**-6)))
            phi_p[i,j] = np.abs(K*count*dx*dy)
        
"""   