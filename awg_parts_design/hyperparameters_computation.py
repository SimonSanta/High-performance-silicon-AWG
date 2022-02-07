# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:35:43 2020

@author: Simon Santacatterina

This script computes the parameters of the AWG according to the paper 
"MUÑOZ et al.: MODELING AND DESIGN OF ARRAYED WAVEGUIDE GRATINGS".

Optical signals operate in C-band, i.e. from 1530 nm to 1565 nm and the 
wavelength of operation is 1550 nm.

Some parameters used in equations have values computed from other softwares (R-Soft)
or given by high-level requirements.
"""
# Initial parameters.
c = 299792458 # speed of light in vacuum

# Network High Level requirement
lambda_init = 1530 * 10**-9 # wavelength where C band starts
lambda_final = 1565 * 10**-9  # wavelength where C band ends
lambda_0 = 1550 * 10**-9  # wavelength of operation for optical signals

f_init = c/lambda_final # frequency where C band starts
f_final = c/lambda_init # frequency where C band ends
f_0 = c/lambda_0 # frequency of operation for optical signals

# Material constant (R-soft)
n_si = 3.47 #Si refractive index
n_eff = 2.304 # Si-waveguide effective refractive index computed by R-Soft
n_fpr = 2.844 # slab refractive index computed by R-soft

"""
1) Computation of the wavelength spacing as function of frequency spacing.

Ideally, Si AWG would have 32-40 OW channels. 
But in this case, the AWG exhibits very poor characteristics and is thus not 
really suitable for applications. Instead, one would prefer using 8 channels, 
400 GHz channel spacing.
"""
f_spacing = 400 * 10**9 # AWG frequency spacing (High level requirement)
lambda_spacing = c*((1/f_0) - 1/(f_0 + f_spacing)) # AWG wavelength spacing (3.2nm)

"""
2) Wavelength band delta_lambda.
For an 8-channel AWG.
"""
n_ch = 8 # n_ch is the number of input channels in the AWG
delta_lambda = lambda_spacing*n_ch # Approximately 25.6 nm

"""
3) Free spectral range (FSR).
"""
fsr = 1.5*delta_lambda # 38.4 nm

"""
4) Focal length of slab waveguide (Lf)
This steps requires some mathematical manipulation:

- alpha = lambda*Lf/ns, is the equivalent to the wavelength focal length product
in Fourier optics propagation. Lf is the length of the FPR and n_fpr its index.

- gamma = f_spacing/d_out, is the frequency spatial dispersion parameter.

- Related by gamma = d_wg*f_0/alpha*m. And we can find Lf.
"""
d_wg = 2 * 10**-6 # arrayed waveguide spacing (High level requirement from R-Soft optimization)
d_out = 2 * 10**-6 # output waveguide spacing (High level requirement from R-Soft optimization)
m_1 = 40 # integer choosen arbitrarily

# compute gamma and alpha from formula given above
gamma = f_spacing/d_out
alpha = (d_wg*f_0)/(m_1*gamma)

Lf = (alpha*n_fpr)/lambda_0 #Lf is the length of the FPR

"""
5) Path difference of arrayed waveguides Delta_l.
The phase delay must be equal to an integer multiple of the design wavelength 
in the waveguides. R-soft effective refractive index is n_eff = 2.304.
"""
lambda_eff = lambda_0/n_eff
m_2 = 40 # m = 40, integer choosen arbitrarily
delta_l = m_2*lambda_eff
